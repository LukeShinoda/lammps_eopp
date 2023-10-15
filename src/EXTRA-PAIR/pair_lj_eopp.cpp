/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include "pair_lj_eopp.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "respa.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */
/*Constructor*/
PairLJEopp::PairLJEopp(LAMMPS *lmp) : Pair(lmp)
{
  respa_enable = 1;
  born_matrix_enable = 1;
  writedata = 1;
}

/* ---------------------------------------------------------------------- */
/*Destructor*/
PairLJEopp::~PairLJEopp()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

void PairLJEopp::compute(int eflag, int vflag)
{
  /*
  
  @TODO: this one is called! let's go baby!
  
  */
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl,oldevdwl, fpair;
  double rsq, r, r2inv, r6inv, forcelj, factor_lj, oldfpair;
  int *ilist, *jlist, *numneigh, **firstneigh;
  bool debug;
  debug = false;
  evdwl = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];
      if (rsq < cutsq[itype][jtype]) {
        /**force calculation*/
          if (debug){
          r2inv = 1.0 / rsq;
          r6inv = r2inv * r2inv * r2inv;
          forcelj = r6inv * (48.0 * epsilon[itype][jtype] * pow(sigma[itype][jtype], 12.0) * r6inv - 24.0 * epsilon[itype][jtype] * pow(sigma[itype][jtype], 6.0));
          fpair = factor_lj * forcelj * r2inv;
          oldfpair = fpair;
          }
          //new force
          r = sqrt(rsq);
          forcelj = lj1[itype][jtype] / pow(r,n1[itype][jtype]);
          //2nd term, chain rule first part
          //printf("the force is %f, we will add %f and then add %f\n", forcelj,lj2[itype][jtype] / pow(r,n2[itype][jtype]) * cos(k[itype][jtype]*r + p[itype][jtype]),lj3[itype][jtype] * k[itype][jtype] / pow(r,n2[itype][jtype]+1.0)*sin(k[itype][jtype]*r + p[itype][jtype]) );
          forcelj +=  lj2[itype][jtype] / pow(r,n2[itype][jtype]) * cos(k[itype][jtype]*r + p[itype][jtype]);
          //2nd term,chain rul second part
          forcelj +=  lj3[itype][jtype] * k[itype][jtype] / pow(r,n2[itype][jtype]+1.0)*sin(k[itype][jtype]*r + p[itype][jtype]);
          fpair = forcelj * (1.0/rsq);
          if (debug && abs(fpair-oldfpair)>0.0001){
            printf("new fpair is %f , old one is %f, difference is %f \n", fpair, oldfpair, fpair-oldfpair );
          }
        //the original implementation is calculating a factor of r^(-12) and r^(-6) for the force
        // multiplied by r^-2 for fpair
        //here the corresponding distances are multiplied back again which makes us reach a factor of r^-13 and r^-7 - as wanted
        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }
        // energy needed 
        if (eflag) {
          if (debug){
              evdwl = r6inv * 4.0 * epsilon[itype][jtype]* pow(sigma[itype][jtype],12.0) * r6inv ;
              evdwl-= r6inv * 4.0 * epsilon[itype][jtype]* pow(sigma[itype][jtype],6.0);
              evdwl-= offset[itype][jtype];
              evdwl *= factor_lj;
              oldevdwl = evdwl;
            }
            //first term
            evdwl = lj3[itype][jtype] / pow(r,n1[itype][jtype]);
            //second term
            evdwl += lj4[itype][jtype] / pow(r,n2[itype][jtype]) * cos(k[itype][jtype]*r + p[itype][jtype]);
            //printf("evdwl is sadly %f. I will even add %f \n", evdwl, offset[itype][jtype]);
            evdwl-= offset[itype][jtype];
        
            if (debug && abs(evdwl-oldevdwl) > 0.0001) {
              printf("evdwl old is %f, new one is %f \n", oldevdwl, evdwl);
            }
          }

        if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
      }
    }
  }
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairLJEopp::compute_inner()
{
  printf("compute inner");
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;
  double rsq, r2inv, r6inv, forcelj, factor_lj, rsw;
  int *ilist, *jlist, *numneigh, **firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum_inner;
  ilist = list->ilist_inner;
  numneigh = list->numneigh_inner;
  firstneigh = list->firstneigh_inner;

  double cut_out_on = cut_respa[0];
  double cut_out_off = cut_respa[1];

  double cut_out_diff = cut_out_off - cut_out_on;
  double cut_out_on_sq = cut_out_on * cut_out_on;
  double cut_out_off_sq = cut_out_off * cut_out_off;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      if (rsq < cut_out_off_sq) {
        r2inv = 1.0 / rsq;
        r6inv = r2inv * r2inv * r2inv;
        jtype = type[j];
        forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
        

        fpair = factor_lj * forcelj * r2inv;
        if (rsq > cut_out_on_sq) {
          rsw = (sqrt(rsq) - cut_out_on) / cut_out_diff;
          fpair *= 1.0 - rsw * rsw * (3.0 - 2.0 * rsw);
        }

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairLJEopp::compute_middle()
{
  printf("compute middle");
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;
  double rsq, r2inv, r6inv, forcelj, factor_lj, rsw;
  int *ilist, *jlist, *numneigh, **firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum_middle;
  ilist = list->ilist_middle;
  numneigh = list->numneigh_middle;
  firstneigh = list->firstneigh_middle;

  double cut_in_off = cut_respa[0];
  double cut_in_on = cut_respa[1];
  double cut_out_on = cut_respa[2];
  double cut_out_off = cut_respa[3];

  double cut_in_diff = cut_in_on - cut_in_off;
  double cut_out_diff = cut_out_off - cut_out_on;
  double cut_in_off_sq = cut_in_off * cut_in_off;
  double cut_in_on_sq = cut_in_on * cut_in_on;
  double cut_out_on_sq = cut_out_on * cut_out_on;
  double cut_out_off_sq = cut_out_off * cut_out_off;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      if (rsq < cut_out_off_sq && rsq > cut_in_off_sq) {
        r2inv = 1.0 / rsq;
        r6inv = r2inv * r2inv * r2inv;
        jtype = type[j];
        forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
        fpair = factor_lj * forcelj * r2inv;
        if (rsq < cut_in_on_sq) {
          rsw = (sqrt(rsq) - cut_in_off) / cut_in_diff;
          fpair *= rsw * rsw * (3.0 - 2.0 * rsw);
        }
        if (rsq > cut_out_on_sq) {
          rsw = (sqrt(rsq) - cut_out_on) / cut_out_diff;
          fpair *= 1.0 + rsw * rsw * (2.0 * rsw - 3.0);
        }

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairLJEopp::compute_outer(int eflag, int vflag)
{
  printf("compute outer");
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double rsq, r2inv, r6inv, forcelj, factor_lj, rsw;
  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double cut_in_off = cut_respa[2];
  double cut_in_on = cut_respa[3];

  double cut_in_diff = cut_in_on - cut_in_off;
  double cut_in_off_sq = cut_in_off * cut_in_off;
  double cut_in_on_sq = cut_in_on * cut_in_on;
  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        if (rsq > cut_in_off_sq) {
          r2inv = 1.0 / rsq;
          r6inv = r2inv * r2inv * r2inv;
          forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);

          fpair = factor_lj * forcelj * r2inv;
          if (rsq < cut_in_on_sq) {
            rsw = (sqrt(rsq) - cut_in_off) / cut_in_diff;
            fpair *= rsw * rsw * (3.0 - 2.0 * rsw);
          }

          f[i][0] += delx * fpair;
          f[i][1] += dely * fpair;
          f[i][2] += delz * fpair;
          if (newton_pair || j < nlocal) {
            f[j][0] -= delx * fpair;
            f[j][1] -= dely * fpair;
            f[j][2] -= delz * fpair;
          }
        }
        //energy needed only
        if (eflag) {
          r2inv = 1.0 / rsq;
          r6inv = r2inv * r2inv * r2inv;
          evdwl = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
          evdwl *= factor_lj;
        }
        //virial needed
        if (vflag) {
          if (rsq <= cut_in_off_sq) {
            r2inv = 1.0 / rsq;
            r6inv = r2inv * r2inv * r2inv;
            forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
            fpair = factor_lj * forcelj * r2inv;
          } else if (rsq < cut_in_on_sq)
            fpair = factor_lj * forcelj * r2inv;
        }
        //either one or the other one
        if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJEopp::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair:setflag");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 0;

  memory->create(cutsq, n, n, "pair:cutsq");

  memory->create(cut, n, n, "pair:cut");
  memory->create(epsilon, n, n, "pair:epsilon");
  memory->create(sigma, n, n, "pair:sigma");
  memory->create(lj1, n, n, "pair:lj1");
  memory->create(lj2, n, n, "pair:lj2");
  memory->create(lj3, n, n, "pair:lj3");
  memory->create(lj4, n, n, "pair:lj4");
  memory->create(offset, n, n, "pair:offset");


  /**
  create new fields*/
  memory->create(c1, n, n, "pair:c1");
  memory->create(c2, n, n, "pair:c2");
  memory->create(n1, n, n, "pair:eta1");
  memory->create(n2, n, n, "pair:eta2");
  memory->create(k, n, n, "pair:kstar");
  memory->create(p, n, n, "pair:phistar");

}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLJEopp::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Illegal pair_style command");

  cut_global = utils::numeric(FLERR, arg[0], false, lmp);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i, j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */
/*This method reads in all pair_coeff starting with arg[0] for the first one.*/
void PairLJEopp::coeff(int narg, char **arg)
{
  if (narg < 4) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  //read in atom types
  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double epsilon_one = utils::numeric(FLERR, arg[2], false, lmp);
  double sigma_one = utils::numeric(FLERR, arg[3], false, lmp);
  double cut_one = cut_global;
  if (narg == 11) cut_one = utils::numeric(FLERR, arg[4], false, lmp);


  //read in all additional parameters needed
  double c_one = utils::numeric(FLERR, arg[5], false, lmp);
  double eta_one = utils::numeric(FLERR, arg[6], false, lmp);
  double c_two =  utils::numeric(FLERR, arg[7], false, lmp);
  double eta_two = utils::numeric(FLERR, arg[8], false, lmp);
  double k_star = utils::numeric(FLERR, arg[9], false, lmp);
  double phi_star = utils::numeric(FLERR, arg[10], false, lmp);
  printf("\n \n hello my friend! I am telling you all additional parameters that are set. are you ready?\n we have %i elements",narg);
  printf("\n C1=%s,n1=%s,C2=%s,n2=%s,k*=%s,phi*=%s \n", arg[5], arg[6], arg[7], arg[8], arg[9], arg[10]);

  //loop through atom types
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;

      //fill new fields
      c1[i][j] = c_one;
      c2[i][j] = c_two;
      n1[i][j] = eta_one;
      n2[i][j] = eta_two;
      k[i][j] = k_star;
      p[i][j] = phi_star;
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJEopp::init_style()
{
  // request regular or rRESPA neighbor list

  int list_style = NeighConst::REQ_DEFAULT;

  if (update->whichflag == 1 && utils::strmatch(update->integrate_style, "^respa")) {
    auto respa = dynamic_cast<Respa *>(update->integrate);
    if (respa->level_inner >= 0) list_style = NeighConst::REQ_RESPA_INOUT;
    if (respa->level_middle >= 0) list_style = NeighConst::REQ_RESPA_ALL;
  }
  neighbor->add_request(this, list_style);

  // set rRESPA cutoffs

  if (utils::strmatch(update->integrate_style, "^respa") &&
      (dynamic_cast<Respa *>(update->integrate))->level_inner >= 0)
    cut_respa = (dynamic_cast<Respa *>(update->integrate))->cutoff;
  else
    cut_respa = nullptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */
/*
called by pair.cpp and saved to variable cut. Furthermore cut is squared. 
This initiates common pre-parameters for LJ potential such that they don't need to be calculated in every iteration
*/
double PairLJEopp::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i], epsilon[j][j], sigma[i][i], sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i], sigma[j][j]);
    cut[i][j] = mix_distance(cut[i][i], cut[j][j]);
  }


  //TODO: maybe reconsider the signs coming from the derivative!!

  //constants from derivative of first term in potential
  // d/dr (c1/r^n1)
  lj1[i][j] =   n1[i][j] * c1[i][j];
  printf("old lj1 is %f, new lj1 is %f \n", 48.0 * epsilon[i][j] * pow(sigma[i][j], 12.0), lj1[i][j] );
  //outter derivative of second term in potential
  //d/dr (c2/r^n2*cos(kr+phi)) --> chain rule lj2 first term, lj3 second term (constants not touched)
  lj2[i][j] =   n2[i][j] * c2[i][j];
  printf("old lj2 is %f, new lj2 is %f \n", 24.0 * epsilon[i][j] * pow(sigma[i][j], 6.0), lj2[i][j] );
  //no derivative, first term
  lj3[i][j] = c1[i][j];
  //no derivative, second term
  printf("old lj3 is %f, new lj3 is %f \n", 4.0 * epsilon[i][j] * pow(sigma[i][j], 12.0), lj3[i][j] );
  lj4[i][j] = c2[i][j];
  printf("old lj4 is %f, new lj4 is %f \n", 4.0 * epsilon[i][j] * pow(sigma[i][j], 6.0), lj4[i][j] );
  /**
  new logic for that part:
  We have V(r) =c1/r^n1 + C2/r^n2*cos(kr+p)
  **/

//TODO: consider new offset
  if (offset_flag && (cut[i][j] > 0.0)) {
    /*double ratio = sigma[i][j] / cut[i][j]; 
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio, 12.0) - pow(ratio, 6.0));
    */
    double ratio1 = c1[i][j]/pow(cut[i][j],n1[i][j]);
    double ratio2 = c2[i][j]/pow(cut[i][j],n2[i][j]);
    printf("ratios are %f,%f\n", ratio1, ratio2);
    offset[i][j] = ratio1 + ratio2;
    printf("offset for %d,%d is %f\n",i,j , offset[i][j]);
  } else
    offset[i][j] = 0.0;

  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  // check interior rRESPA cutoff

  if (cut_respa && cut[i][j] < cut_respa[3])
    error->all(FLERR, "Pair cutoff < Respa interior cutoff");

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;

    double count[2], all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count, all, 2, MPI_DOUBLE, MPI_SUM, world);

    double sig2 = sigma[i][j] * sigma[i][j];
    double sig6 = sig2 * sig2 * sig2;
    double rc3 = cut[i][j] * cut[i][j] * cut[i][j];
    double rc6 = rc3 * rc3;
    double rc9 = rc3 * rc6;
    double prefactor = 8.0 * MY_PI * all[0] * all[1] * epsilon[i][j] * sig6 / (9.0 * rc9);
    etail_ij = prefactor * (sig6 - 3.0 * rc6);
    ptail_ij = 2.0 * prefactor * (2.0 * sig6 - 3.0 * rc6);
  }

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJEopp::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j], sizeof(double), 1, fp);
        fwrite(&sigma[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJEopp::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i, j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR, &epsilon[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &sigma[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&epsilon[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&sigma[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJEopp::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global, sizeof(double), 1, fp);
  fwrite(&offset_flag, sizeof(int), 1, fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
  fwrite(&tail_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJEopp::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    utils::sfread(FLERR, &cut_global, sizeof(double), 1, fp, nullptr, error);
    utils::sfread(FLERR, &offset_flag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &tail_flag, sizeof(int), 1, fp, nullptr, error);
  }
  MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&offset_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&tail_flag, 1, MPI_INT, 0, world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairLJEopp::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++) fprintf(fp, "%d %g %g\n", i, epsilon[i][i], sigma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairLJEopp::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g\n", i, j, epsilon[i][j], sigma[i][j], cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairLJEopp::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                         double /*factor_coul*/, double factor_lj, double &fforce)
{
  double r2inv, r6inv, forcelj, philj;

  r2inv = 1.0 / rsq;
  r6inv = r2inv * r2inv * r2inv;
  forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
  fforce = factor_lj * forcelj * r2inv;

  philj = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
  return factor_lj * philj;
}

/* ---------------------------------------------------------------------- */

void PairLJEopp::born_matrix(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                            double /*factor_coul*/, double factor_lj, double &dupair,
                            double &du2pair)
{
  
  double rinv, r2inv, r6inv, du, du2;

  r2inv = 1.0 / rsq;
  rinv = sqrt(r2inv);
  r6inv = r2inv * r2inv * r2inv;

  // Reminder: lj1 = 48*e*s^12, lj2 = 24*e*s^6
  // so dupair = -forcelj/r = -fforce*r (forcelj from single method)

  du = r6inv * rinv * (lj2[itype][jtype] - lj1[itype][jtype] * r6inv);
  du2 = r6inv * r2inv * (13 * lj1[itype][jtype] * r6inv - 7 * lj2[itype][jtype]);

  dupair = factor_lj * du;
  du2pair = factor_lj * du2;
}

/* ---------------------------------------------------------------------- */

void *PairLJEopp::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str, "epsilon") == 0) return (void *) epsilon;
  if (strcmp(str, "sigma") == 0) return (void *) sigma;
  return nullptr;
}