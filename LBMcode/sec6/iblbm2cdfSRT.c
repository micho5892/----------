// cylindrical Couette flow
// Immersed Boundary-Lattice Boltzmann Method ( Direct Forcing Method )
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
// cx  : discrete velocity (x-axis)
// cy  : discrete velocity (y-axis)
// tau : relaxation time
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
# define pd 100

int main(void)
{
  FILE   *fp;
  int    nx = 50, ny = 50, time = 0, loop1, loop2;
  int    i, j, k, n, m, in, jn, ip, im, jp, jm;
  double rho[DIM][DIM], u[DIM][DIM], v[DIM][DIM], ut[DIM][DIM], ua[DIM][DIM];
  double fx[DIM][DIM], fy[DIM][DIM];
  double f[9][DIM][DIM], f0[9][DIM][DIM], ftmp[9][DIM][DIM], cx[9], cy[9];
  double  u0 = 0.01, umax, umin, tmp, u2, nu, tau, err, tmp1, tmp2, tmp3;

  double xp[ed], yp[ed], rp[ed];
  double xe[ed][pd], ye[ed][pd], fxe[ed][pd], fye[ed][pd];
  double uet[ed][pd], vet[ed][pd], ue[ed][pd], ve[ed][pd];
  int    np, ne[ed];
  char   a[DIM][DIM];

  // initial condition
  tau = 0.6;
//  tau = 10.0;
  nu = (tau - 0.5)/3.0;

// particle
  np = 2;
  rp[0] = 70.0/200.0*(double)nx;
  rp[1] = 45.0/200.0*(double)nx;
  ne[0] = (int)(2.0 * M_PI * rp[0] * 0.5);
  ne[1] = (int)(2.0 * M_PI * rp[1] * 0.5);

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

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
      f[k][i][j] = f[k][i][j] - (f[k][i][j] - f0[k][i][j])/tau;
    } } }

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      for(k = 1; k <= 4; k++){
        f[k][i][j] = f[k][i][j] + (fx[i][j]*cx[k] + fy[i][j]*cy[k])/3.0;
      }
      for(k = 5; k <= 8; k++){
        f[k][i][j] = f[k][i][j] + (fx[i][j]*cx[k] + fy[i][j]*cy[k])/12.0;
      }
    } }

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

  fp = fopen("datau","w");
    for(j = 1; j <= ny-1; j++){ for(i = 1; i <= nx-1; i++){
      fprintf(fp," %10.8e", u[i][j]);
    }
  fprintf(fp,"\n");
  } 
  fclose(fp);

  fp = fopen("datav","w");
    for(j = 1; j <= ny-1; j++){ for(i = 1; i <= nx-1; i++){
      fprintf(fp," %10.8e", v[i][j]);
    }
  fprintf(fp,"\n");
  } 
  fclose(fp);

  return 0;
}
