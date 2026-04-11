// cylindrical Couette flow
// Immersed Boundary-Lattice Boltzmann Method ( Implicit Correction Method ) with the TRT collision operator
// The source code is written in C programming language.
// Copyright @2021 Takeshi Seta All Rights Reserved.
//
// 6  2  5
//    |
// 3--0--1
//    |
// 7  4  8
//
// rho : density
// u   : horizontal velocity component (time = n)
// v   : vertical velocity component (time = n)
// ua   : clockwise tangential velocity (analytical solution)
// ut   : clockwise tangential velocity (numerical solution)
// u0  :  velocity of the inner cylinder
//
// nx  : number of grid points (x-axis)
// ny  : number of grid points (y-axis)
// f   : distribution function
// f0  : equilibrium distribution function
// fp  : distribution function for even part
// fm  : distribution function for odd part
// f0  : equilibrium distribution function
// f0p : equilibrium distribution function for even part
// f0m : equilibrium distribution function for odd part
// fi  : forcing term
// fip : forcing term for even part
// fim : forcing term for odd part
// cx  : discrete velocity (x-axis)
// cy  : discrete velocity (y-axis)
// taup: relaxation time for even part
// taum: relaxation time for odd part
// nu  : kinematic viscosity
//
// np  : number of particles
// rp  : radius of particle
// xp  : position of particles (x-axis)
// yp  : position of particles (y-axis)
//
// ne  : number of Lagrangian points
// xe  : positon of the Lagrangian points (x-axis)
// ye  : positon of the Lagrangian points (y-axis)
// fxe : force acting on Lagrangian points (x-axis)
// fye : force acting on Lagrangian points (x-axis)
// ue  : horizontal velocity component of the Lagrangian points
// ve  : vertical velocity component  of the Lagrangian points
// uet : interpolated fluid velocity on the Lagrangian points (x-axis)
// vet : interpolated fluid velocity on the Lagrangian points (y-axis)

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

# define DIM 51 
# define ed 2
# define pd 25

int main(void)
{
  FILE   *fpp;
  int    nx = 50, ny = 50, time = 0, loop1, loop2;
  int    i, j, k, n, m, in, jn, ip, im, jp, jm;
  double rho[DIM][DIM], u[DIM][DIM], v[DIM][DIM], ut[DIM][DIM], ua[DIM][DIM];
  double fx[DIM][DIM], fy[DIM][DIM];
  double f[9][DIM][DIM], f0[9][DIM][DIM], ftmp[9][DIM][DIM], cx[9], cy[9];
  double fp[9][DIM][DIM], fm[9][DIM][DIM], f0p[9][DIM][DIM], f0m[9][DIM][DIM];
  double fi[9][DIM][DIM];
  double fip[9][DIM][DIM];
  double fim[9][DIM][DIM];
  double u0 = 0.01, umax, umin, tmp, u2, nu, taup, taum, err, tmp1, tmp2, tmp3;

  double xp[ed], yp[ed], rp[ed];
  double xe[ed][pd], ye[ed][pd], fxe[ed][pd], fye[ed][pd];
  double uet[ed][pd], vet[ed][pd], ue[ed][pd], ve[ed][pd];
  double mm[ed][pd][pd];
  double b[pd], r;
  int    np, ne[ed];
  char   a[DIM][DIM];

  // initial condition
  taup = 0.6;
  taup = 10.0;
  nu = (taup - 0.5)/3.0;
//.. slip velocity (us/u0 = 0)
//  taum = taup;
  taum = (4.0*taup + 7.0)/(8.0*taup - 4.0);

// particle
  np = 2;
  rp[0] = 70.0/200.0*(double)nx;
  rp[1] = 45.0/200.0*(double)nx;
  ne[0] = (int)(2.0 * M_PI * rp[0] * 0.2);
  ne[1] = (int)(2.0 * M_PI * rp[1] * 0.2);

  xp[0] = 0.5*(double)nx;
  yp[0] = 0.5*(double)nx;
  xp[1] = 0.5*(double)nx;
  yp[1] = 0.5*(double)nx;

  for(n = 0; n < np; n++) { for(m = 0; m <ne[n] ; m++) {
     xe[n][m] = xp[n] + rp[n]*cos(2.0*M_PI*(double)m/(double)ne[n]);
     ye[n][m] = yp[n] + rp[n]*sin(2.0*M_PI*(double)m/(double)ne[n]);
    fxe[n][m] = 0.0; fye[n][m] = 0.0; ue[n][m] = 0.0; ve[n][m] = 0.0;
  } } 
  printf("np=%d, ne1=%d, ne2=%d, rp1=%f, rp2=%f, \n", np, ne[0], ne[1], rp[0], rp[1]);

//.. velocity correction method ( matrix formulation )
  for(n = 0; n < np; n++) { for(in = 0; in <ne[n]; in++) { for(jn = 0; jn <ne[n]; jn++) {
    mm[n][in][jn] = 0.0;
  }  }  }

  for(n = 0; n < np; n++) { for(in = 0; in <ne[n]; in++) { for(jn = 0; jn <ne[n]; jn++) {
    for(i = (int)xe[n][in] - 3; i < (int)xe[n][in] + 3; i++){
    for(j = (int)ye[n][in] - 3; j < (int)ye[n][in] + 3; j++){
      tmp1 = fabs(xe[n][in] - (double)i);
      tmp2 = fabs(ye[n][in] - (double)j);
      if(tmp1 <= 2.0){
        tmp3 = (1.0 + cos(M_PI*tmp1/2.0))/4.0;
      } else {
        tmp3 = 0.0;
      }
      if(tmp2 <= 2.0){
        tmp3 = (1.0 + cos(M_PI*tmp2/2.0))/4.0*tmp3;
      } else {
        tmp3 = 0.0;
      }

      tmp1 = fabs(xe[n][jn] - (double)i);
      tmp2 = fabs(ye[n][jn] - (double)j);
      if(tmp1 <= 2.0){
        tmp3 = (1.0 + cos(M_PI*tmp1/2.0))/4.0*tmp3;
      } else {
        tmp3 = 0.0;
      }
      if(tmp2 <= 2.0){
        tmp3 = (1.0 + cos(M_PI*tmp2/2.0))/4.0*tmp3;
      } else {
        tmp3 = 0.0;
      }
      mm[n][in][jn] = mm[n][in][jn] + tmp3;
    } }
  } } }

  for(n = 0; n < np; n++) { for(in = 0; in <ne[n]; in++) { for(jn = 0; jn <ne[n]; jn++) {
    mm[n][in][jn] = mm[n][in][jn]*2.0*M_PI*rp[n]/(double)ne[n];
  } } }

  for(m = 0; m <ne[1] ; m++) {
     ue[1][m] =  u0*sin(2.*M_PI*(double)m/(double)ne[1]);
     ve[1][m] = -u0*cos(2.*M_PI*(double)m/(double)ne[1]);
  }

//..  exact solution
  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ ua[i][j] = 0.0; } }

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    tmp = sqrt((double)(i - nx/2)*(double)(i - nx/2) +
             + (double)(j - ny/2)*(double)(j - ny/2) );
    ua[i][j] = u0*(tmp/rp[0] - rp[0]/tmp)/(rp[1]/rp[0] - rp[0]/rp[1]);
    if(tmp  <= rp[1] ){ ua[i][j] = u0; }
    if(tmp  >= rp[0] ){ ua[i][j] = 0.0;}
  } }

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    u[i][j] = 0.0; v[i][j] = 0.0; rho[i][j] = 1.0;
  } }

  cx[0] = 0.0; cy[0] = 0.0;
  cx[1] = 1.0; cy[1] = 0.0;  cx[2] = 0.0; cy[2] = 1.0;
  cx[3] =-1.0; cy[3] = 0.0;  cx[4] = 0.0; cy[4] =-1.0;
  cx[5] = 1.0; cy[5] = 1.0;  cx[6] =-1.0; cy[6] = 1.0;
  cx[7] =-1.0; cy[7] =-1.0;  cx[8] = 1.0; cy[8] =-1.0;

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    u2 = u[i][j]*u[i][j] + v[i][j]*v[i][j];      
    f0[0][i][j] = rho[i][j]*(1.0 -3.0/2.0*u2)*4.0/9.0;
    for(k = 1; k <= 4; k++){
      tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
      f0[k][i][j] = rho[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/9.0;
    }
    for(k = 5; k <= 8; k++){
      tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
      f0[k][i][j] = rho[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
    }
  } }

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
    f[k][i][j] = f0[k][i][j];
  } } }

// calculation start
  for(loop1 = 0; loop1 < 20; loop1++){
  for(loop2 = 0; loop2 < 100; loop2++){
    time++;

  // collision
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      u2 = u[i][j]*u[i][j] + v[i][j]*v[i][j];      
      f0[0][i][j] = rho[i][j]*(1.0 -3.0/2.0*u2)*4.0/9.0;
      for(k = 1; k <= 4; k++){
        tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
        f0[k][i][j] = rho[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/9.0;
      }  
      for(k = 5; k <= 8; k++){
        tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
        f0[k][i][j] = rho[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
      }
    } }

//.. Two Relaxation Time
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
       fp[0][i][j] = f[0][i][j]; f0p[0][i][j] = f0[0][i][j];
       fm[0][i][j] = 0.0       ; f0m[0][i][j] = 0.0;

       fp[1][i][j] = (f[1][i][j] + f[3][i][j])*0.5;
       fp[2][i][j] = (f[2][i][j] + f[4][i][j])*0.5;
       fp[3][i][j] = (f[3][i][j] + f[1][i][j])*0.5;
       fp[4][i][j] = (f[4][i][j] + f[2][i][j])*0.5;
       fp[5][i][j] = (f[5][i][j] + f[7][i][j])*0.5;
       fp[6][i][j] = (f[6][i][j] + f[8][i][j])*0.5;
       fp[7][i][j] = (f[7][i][j] + f[5][i][j])*0.5;
       fp[8][i][j] = (f[8][i][j] + f[6][i][j])*0.5;

       f0p[1][i][j] = (f0[1][i][j] + f0[3][i][j])*0.5;
       f0p[2][i][j] = (f0[2][i][j] + f0[4][i][j])*0.5;
       f0p[3][i][j] = (f0[3][i][j] + f0[1][i][j])*0.5;
       f0p[4][i][j] = (f0[4][i][j] + f0[2][i][j])*0.5;
       f0p[5][i][j] = (f0[5][i][j] + f0[7][i][j])*0.5;
       f0p[6][i][j] = (f0[6][i][j] + f0[8][i][j])*0.5;
       f0p[7][i][j] = (f0[7][i][j] + f0[5][i][j])*0.5;
       f0p[8][i][j] = (f0[8][i][j] + f0[6][i][j])*0.5;

       fm[1][i][j] = (f[1][i][j] - f[3][i][j])*0.5;
       fm[2][i][j] = (f[2][i][j] - f[4][i][j])*0.5;
       fm[3][i][j] = (f[3][i][j] - f[1][i][j])*0.5;
       fm[4][i][j] = (f[4][i][j] - f[2][i][j])*0.5;
       fm[5][i][j] = (f[5][i][j] - f[7][i][j])*0.5;
       fm[6][i][j] = (f[6][i][j] - f[8][i][j])*0.5;
       fm[7][i][j] = (f[7][i][j] - f[5][i][j])*0.5;
       fm[8][i][j] = (f[8][i][j] - f[6][i][j])*0.5;

       f0m[1][i][j] = (f0[1][i][j] - f0[3][i][j])*0.5;
       f0m[2][i][j] = (f0[2][i][j] - f0[4][i][j])*0.5;
       f0m[3][i][j] = (f0[3][i][j] - f0[1][i][j])*0.5;
       f0m[4][i][j] = (f0[4][i][j] - f0[2][i][j])*0.5;
       f0m[5][i][j] = (f0[5][i][j] - f0[7][i][j])*0.5;
       f0m[6][i][j] = (f0[6][i][j] - f0[8][i][j])*0.5;
       f0m[7][i][j] = (f0[7][i][j] - f0[5][i][j])*0.5;
       f0m[8][i][j] = (f0[8][i][j] - f0[6][i][j])*0.5;
    } }

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
      f[k][i][j] =  f[k][i][j] - (fp[k][i][j] - f0p[k][i][j])/taup
                               - (fm[k][i][j] - f0m[k][i][j])/taum;
    } } }


    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      u2 = u[i][j]*fx[i][j] + v[i][j]*fy[i][j];

      fi[0][i][j] = -4.0/3.0*u2;

     for(k = 1; k <= 4; k++){
       tmp = cx[k]*fx[i][j] + cy[k]*fy[i][j];

       tmp1 = u[i][j]*fx[i][j]*cx[k]*cx[k]
            + u[i][j]*fy[i][j]*cx[k]*cy[k]
            + v[i][j]*fx[i][j]*cy[k]*cx[k]
            + v[i][j]*fy[i][j]*cy[k]*cy[k];
       fi[k][i][j] = (3.0*tmp + 9.0*tmp1 - 3.0*u2)/9.0;
      }
     for(k = 5; k <= 8; k++){
       tmp = cx[k]*fx[i][j] + cy[k]*fy[i][j];
       tmp1 = u[i][j]*fx[i][j]*cx[k]*cx[k]
            + u[i][j]*fy[i][j]*cx[k]*cy[k]
            + v[i][j]*fx[i][j]*cy[k]*cx[k]
            + v[i][j]*fy[i][j]*cy[k]*cy[k];
       fi[k][i][j] = (3.0*tmp + 9.0*tmp1 - 3.0*u2)/36.0;
      }
    } }

//.. Two Relaxation Time
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
       fip[0][i][j] = fi[0][i][j];

       fip[1][i][j] = (fi[1][i][j] + fi[3][i][j])*0.5;
       fip[2][i][j] = (fi[2][i][j] + fi[4][i][j])*0.5;
       fip[3][i][j] = (fi[3][i][j] + fi[1][i][j])*0.5;
       fip[4][i][j] = (fi[4][i][j] + fi[2][i][j])*0.5;
       fip[5][i][j] = (fi[5][i][j] + fi[7][i][j])*0.5;
       fip[6][i][j] = (fi[6][i][j] + fi[8][i][j])*0.5;
       fip[7][i][j] = (fi[7][i][j] + fi[5][i][j])*0.5;
       fip[8][i][j] = (fi[8][i][j] + fi[6][i][j])*0.5;

       fim[0][i][j] = 0.0;
       fim[1][i][j] = (fi[1][i][j] - fi[3][i][j])*0.5;
       fim[2][i][j] = (fi[2][i][j] - fi[4][i][j])*0.5;
       fim[3][i][j] = (fi[3][i][j] - fi[1][i][j])*0.5;
       fim[4][i][j] = (fi[4][i][j] - fi[2][i][j])*0.5;
       fim[5][i][j] = (fi[5][i][j] - fi[7][i][j])*0.5;
       fim[6][i][j] = (fi[6][i][j] - fi[8][i][j])*0.5;
       fim[7][i][j] = (fi[7][i][j] - fi[5][i][j])*0.5;
       fim[8][i][j] = (fi[8][i][j] - fi[6][i][j])*0.5;
    } }

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
      f[k][i][j] =  f[k][i][j] + (1.0 - 0.5/taup)*fip[k][i][j]
                               + (1.0 - 0.5/taum)*fim[k][i][j];
    } } }

//    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
//      f[k][i][j] = f[k][i][j] + fi[k][i][j];
//    } } }

    // propagation 
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
      ftmp[k][i][j] = f[k][i][j];
    } } }

    for(i = 1; i <= nx-1; i++){ for(j = 1; j <= ny-1; j++){ for(k = 0; k <= 8; k++){
      in = i + (int)cx[k]; jn = j + (int)cy[k];
      if(in == nx + 1){in =  0;}
      if(in ==    - 1){in = nx;}
      if(jn == ny + 1){jn =  0;}
      if(jn ==    - 1){jn = ny;}
      f[k][in][jn] = ftmp[k][i][j];
    } } }

    // physics
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      rho[i][j] = f[0][i][j]; u[i][j] = 0; v[i][j] = 0;
      for( k = 1; k <= 8; k++){
        rho[i][j] = rho[i][j] + f[k][i][j];
          u[i][j] =   u[i][j] + f[k][i][j]*cx[k];
          v[i][j] =   v[i][j] + f[k][i][j]*cy[k];
      } 
      u[i][j] = u[i][j]/rho[i][j];
      v[i][j] = v[i][j]/rho[i][j];
    } }

// immersed boundary method
    for(n = 0; n < np; n++) { for(m = 0; m <ne[n] ; m++) {
      uet[n][m] = 0.0; vet[n][m] = 0.0;
    } }

    for(n = 0; n < np; n++) { for(m = 0; m <ne[n] ; m++) {
      for(i = (int)xe[n][m] - 3; i < (int)xe[n][m] + 3; i++){
      for(j = (int)ye[n][m] - 3; j < (int)ye[n][m] + 3; j++){
        tmp1 = fabs(xe[n][m] - (double)i);
        tmp2 = fabs(ye[n][m] - (double)j);

        if(tmp1 <= 2.0){
          tmp3 = (1.0 + cos(M_PI*tmp1/2.0))/4.0;
        } else {
          tmp3 = 0.0;
        }
        if(tmp2 <= 2.0){
          tmp3 = (1.0 + cos(M_PI*tmp2/2.0))/4.0*tmp3;
        } else {
          tmp3 = 0.0;
        }

        uet[n][m] += u[i][j]*tmp3;
        vet[n][m] += v[i][j]*tmp3;
      } }
    } }

    for(n = 0; n < np; n++) { for(m = 0; m <ne[n] ; m++) {
      fxe[n][m] = 0.0; fye[n][m] = 0.0; 
    } }

    for(n = 0; n < np; n++) { for(m = 0; m <ne[n] ; m++) {
      fxe[n][m] = ue[n][m] - uet[n][m];
      fye[n][m] = ve[n][m] - vet[n][m];
    } }
//gauss method
    for(in = 0; in < 2; in++) {

      for(n = 0; n < np; n++) {
        for (k = 0; k < ne[n]; k++){

          if(in == 0) {
            for (k = 0; k < ne[n]; k++){
              b[k] = fxe[n][k];
            }
          }else{
            for (k = 0; k < ne[n]; k++){
              b[k] = fye[n][k];
            }
          }

          for (i = k + 1; i < ne[n]; i++){
            r = mm[n][i][k]/ mm[n][k][k];

            for (j = k; j < ne[n]; j++){
              mm[n][i][j] -= mm[n][k][j] * r;
            }
            b[i] -= b[k] * r;
          }
        }

        for (i = ne[n] -1; i >= 0; i--){
          for (j = i + 1; j < ne[n]; j++){
            b[i] -= mm[n][i][j] * b[j];
          }
          b[i] /= mm[n][i][i];
        }

        if(in == 0) {
          for (k = 0; k < ne[n]; k++){
            fxe[n][k] = b[k];
          }
        }else{
          for (k = 0; k < ne[n]; k++){
            fye[n][k] = b[k];
          }
        }
      }
    }


// Spreading: force (fxpe, fype) --> (fx, fy)
    for(i = 0; i < nx; i++) { for(j = 0; j < ny; j++) {
      fx[i][j] = 0.0; fy[i][j] = 0.0;
    } }

    for(n = 0; n < np; n++) { for(m = 0; m <ne[n] ; m++) {
      for(i = (int)xe[n][m] - 3; i < (int)xe[n][m] + 3; i++){
      for(j = (int)ye[n][m] - 3; j < (int)ye[n][m] + 3; j++){

        tmp1 = fabs(xe[n][m] - (double)i);
        tmp2 = fabs(ye[n][m] - (double)j);

        if(tmp1 <= 2.0){
          tmp3 = (1.0 + cos(M_PI*tmp1/2.0))/4.0;
        } else {
          tmp3 = 0.0;
        }
        if(tmp2 <= 2.0){
          tmp3 = (1.0 + cos(M_PI*tmp2/2.0))/4.0*tmp3;
        } else {
          tmp3 = 0.0;
        }

        fx[i][j] += fxe[n][m] * tmp3 * 2.0f*M_PI*rp[n]/(double)ne[n];
        fy[i][j] += fye[n][m] * tmp3 * 2.0f*M_PI*rp[n]/(double)ne[n];
      } }
    } }

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      u[i][j] = u[i][j] + fx[i][j];
      v[i][j] = v[i][j] + fy[i][j];
    } }

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      fx[i][j] = 2.0*fx[i][j];
      fy[i][j] = 2.0*fy[i][j];
    } }


  } //loop2
  
//  clockwise tangential velocity
  for(i = 0; i < nx; i++) { for(j = 0; j < ny; j++) { ut[i][j] = 0.0; } }
  for(i = 0; i < nx; i++) { for(j = 0; j < ny; j++) {
    tmp = sqrt( (double)(i - nx/2)*(double)(i - nx/2)
              + (double)(j - ny/2)*(double)(j - ny/2));
    ut[i][j] = (u[i][j]*(double)(j - ny/2) - v[i][j]*(double)(i - nx/2))/tmp;
  } }

  err = 0.0; umin = 0.0; umax = 0.0;
  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    u2 = sqrt((double)(i - nx/2)*(double)(i - nx/2) +
            + (double)(j - ny/2)*(double)(j - ny/2) );

    if(rp[0] >= u2 && rp[1] <= u2){
      umax = umax + (ut[i][j] - ua[i][j])*(ut[i][j] - ua[i][j]);
      umin = umin + ua[i][j]*ua[i][j];
    }
  } }
  err = sqrt(umax/umin);
  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    a[i][j]='0';
  } }

  umax = -999.9; umin = 999.9;
  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = sqrt(u[i][j]*u[i][j] + v[i][j]*v[i][j]);
    if(tmp >= umax){umax = tmp;}
    if(tmp <= umin){umin = tmp;}
  } }

  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = sqrt(pow(u[i][j],2) + pow(v[i][j],2));
    if(tmp <=  umax*1.0            ){a[i][j] = '9';}
    if(tmp <= (umax*0.9 + umin*0.1)){a[i][j] = '8';}
    if(tmp <= (umax*0.8 + umin*0.2)){a[i][j] = '7';}
    if(tmp <= (umax*0.7 + umin*0.3)){a[i][j] = '6';}
    if(tmp <= (umax*0.6 + umin*0.4)){a[i][j] = '5';}
    if(tmp <= (umax*0.5 + umin*0.5)){a[i][j] = '4';}
    if(tmp <= (umax*0.4 + umin*0.6)){a[i][j] = '3';}
    if(tmp <= (umax*0.3 + umin*0.7)){a[i][j] = '2';}
    if(tmp <= (umax*0.2 + umin*0.8)){a[i][j] = '1';}
    if(tmp <= (umax*0.1 + umin*0.9)){a[i][j] = '0';}
  } }

  for(j = ny - 1; j >= 1; j = j - 2){ for(i = 1; i <= nx - 1; i = i + 2){
    printf("%c",a[i][j]);
    }
    printf("\n");
  }

  printf("%d\t,err:%f, max:%f,min:%f\n",time, err, umax, umin);

  } //loop1

  fpp = fopen("datau","w");
    for(j = 1; j <= ny-1; j++){ for(i = 1; i <= nx-1; i++){
      fprintf(fpp," %10.8e", u[i][j]);
    }
  fprintf(fpp,"\n");
  } 
  fclose(fpp);

  fpp = fopen("datav","w");
    for(j = 1; j <= ny-1; j++){ for(i = 1; i <= nx-1; i++){
      fprintf(fpp," %10.8e", v[i][j]);
    }
  fprintf(fpp,"\n");
  } 
  fclose(fpp);

  return 0;
}
