// cavity flow 
// Lattice Boltzmann Method. The source code is written c programming language.
// The source code is written in C programming language.
// Copyright @2021 Takeshi Seta All Rights Reserved.
//
// 6  2  5
//    |
// 3--0--1
//    |
// 7  4  8
//
// rho: density
// u  : horizontal velocity component (time = n)
// v  : vertical velocity component (time = n)
// un : horizontal velocity component (time = n -1)
// vn : vertical velocity component (time = n - 1)
// u0 : horizontal velocity component of the top wall
// sf : stream function
// nx : number of grid points (x-axis)
// ny : number of grid points (y-axis)
// f  : distribution function
// f0 : equilibrium distribution function
// cx : discrete velocity (x-axis)
// cy : discrete velocity (y-axis)
// tau: relaxation time
// nu : kinematic viscosity
// re : Reynolds number
// mc : transformation matrix
// nc : shift matrix
// mi : inverse matrix of mc
// ni : inverse matrix of nc
// sc : relaxation matrix
// flag = 1 (SRT colision)
// flag = 2 (MRT colision)
// flag = 3 (central moment) 

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

# define DIM 55

int main(void)
{
  FILE   *fp;
  int    nx = 51, ny = 51, time = 0, loop1, loop2;
  int    i, j, k, m, in, jn, ip, im, jp, jm, flag;
  double mc[9][9], nc[9][9], mi[9][9], ni[9][9], sc[9][9];
  double t[9][DIM][DIM], t0[9][DIM][DIM], tn[9][DIM][DIM], tn0[9][DIM][DIM], fm[9][DIM][DIM];
  double rho[DIM][DIM], u[DIM][DIM], v[DIM][DIM], un[DIM][DIM], vn[DIM][DIM], sf[DIM][DIM];
  double f[9][DIM][DIM], f0[9][DIM][DIM], ftmp[9][DIM][DIM], cx[9], cy[9];
  double u0 = 0.100;
  double smax, smin, tmp, u2, nu, norm, tau, re;
  char   a[DIM][DIM];

//  flag = 1; // SRT colision
//  flag = 2; // MRT colision
  flag = 3; // central moment 

  // initial condition
//  re =  100; // SRT, MRT, CM
//  re =  500; // MRT, CM
//  re = 1000; // MRT, CM
  re = 5000; // CM
  nu = u0*(double)(nx - 1)/re;
  tau = 3.0*nu + 0.5;

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    u[i][j] = 0.0; v[i][j] = 0.0; un[i][j] = 0.0; vn[i][j] = 0.0;
    rho[i][j] = 1.0;
  } }

  cx[0] = 0.0; cy[0] = 0.0; cx[1] = 1.0; cy[1] = 0.0; cx[2] = 0.0; cy[2] = 1.0;
  cx[3] =-1.0; cy[3] = 0.0; cx[4] = 0.0; cy[4] =-1.0; cx[5] = 1.0; cy[5] = 1.0;
  cx[6] =-1.0; cy[6] = 1.0; cx[7] =-1.0; cy[7] =-1.0; cx[8] = 1.0; cy[8] =-1.0;

  if(flag == 2){
// MRT collision
    mc[0][0] = 1.0; mc[0][1] = 1.0; mc[0][2] = 1.0;
    mc[0][3] = 1.0; mc[0][4] = 1.0; mc[0][5] = 1.0;
    mc[0][6] = 1.0; mc[0][7] = 1.0; mc[0][8] = 1.0;

    mc[1][0] =-4.0; mc[1][1] =-1.0; mc[1][2] =-1.0;
    mc[1][3] =-1.0; mc[1][4] =-1.0; mc[1][5] = 2.0;
    mc[1][6] = 2.0; mc[1][7] = 2.0; mc[1][8] = 2.0;

    mc[2][0] = 4.0; mc[2][1] =-2.0; mc[2][2] =-2.0;
    mc[2][3] =-2.0; mc[2][4] =-2.0; mc[2][5] = 1.0;
    mc[2][6] = 1.0; mc[2][7] = 1.0; mc[2][8] = 1.0;

    mc[3][0] = 0.0; mc[3][1] = 1.0; mc[3][2] = 0.0;
    mc[3][3] =-1.0; mc[3][4] = 0.0; mc[3][5] = 1.0;
    mc[3][6] =-1.0; mc[3][7] =-1.0; mc[3][8] = 1.0;

    mc[4][0] = 0.0; mc[4][1] =-2.0; mc[4][2] = 0.0;
    mc[4][3] = 2.0; mc[4][4] = 0.0; mc[4][5] = 1.0;
    mc[4][6] =-1.0; mc[4][7] =-1.0; mc[4][8] = 1.0;

    mc[5][0] = 0.0; mc[5][1] = 0.0; mc[5][2] = 1.0; 
    mc[5][3] = 0.0; mc[5][4] =-1.0; mc[5][5] = 1.0; 
    mc[5][6] = 1.0; mc[5][7] =-1.0; mc[5][8] =-1.0;
 
    mc[6][0] = 0.0; mc[6][1] = 0.0; mc[6][2] =-2.0; 
    mc[6][3] = 0.0; mc[6][4] = 2.0; mc[6][5] = 1.0;
    mc[6][6] = 1.0; mc[6][7] =-1.0; mc[6][8] =-1.0;

    mc[7][0] = 0.0; mc[7][1] = 1.0; mc[7][2] =-1.0;
    mc[7][3] = 1.0; mc[7][4] =-1.0; mc[7][5] = 0.0;
    mc[7][6] = 0.0; mc[7][7] = 0.0; mc[7][8] = 0.0;
  
    mc[8][0] = 0.0; mc[8][1] = 0.0; mc[8][2] = 0.0;
    mc[8][3] = 0.0; mc[8][4] = 0.0; mc[8][5] = 1.0;
    mc[8][6] =-1.0; mc[8][7] = 1.0; mc[8][8] =-1.0;

// Inverse matrix of MRT collison
    mi[0][0] = 1.0/9.0 ; mi[0][1] =-1.0/9.0 ; mi[0][2] = 1.0/9.0 ;
    mi[0][3] = 0.0     ; mi[0][4] = 0.0     ; mi[0][5] = 0.0     ;
    mi[0][6] = 0.0     ; mi[0][7] = 0.0     ; mi[0][8] = 0.0     ;

    mi[1][0] = 1.0/9.0 ; mi[1][1] =-1.0/36.0; mi[1][2] =-1.0/18.0;
    mi[1][3] = 1.0/6.0 ; mi[1][4] =-1.0/6.0 ; mi[1][5] = 0.0     ;
    mi[1][6] = 0.0     ; mi[1][7] = 1.0/4.0 ; mi[1][8] = 0.0     ;

    mi[2][0] = 1.0/9.0 ; mi[2][1] =-1.0/36.0; mi[2][2] =-1.0/18.0;
    mi[2][3] = 0.0     ; mi[2][4] = 0.0     ; mi[2][5] = 1.0/6.0 ;
    mi[2][6] =-1.0/6.0 ; mi[2][7] =-1.0/4.0 ; mi[2][8] = 0.0     ;

    mi[3][0] = 1.0/9.0 ; mi[3][1] =-1.0/36.0; mi[3][2] =-1.0/18.0;
    mi[3][3] =-1.0/6.0 ; mi[3][4] = 1.0/6.0 ; mi[3][5] = 0.0     ;
    mi[3][6] = 0.0     ; mi[3][7] = 1.0/4.0 ; mi[3][8] = 0.0     ;

    mi[4][0] = 1.0/9.0 ; mi[4][1] =-1.0/36.0; mi[4][2] =-1.0/18.0;
    mi[4][3] = 0.0     ; mi[4][4] = 0.0     ; mi[4][5] =-1.0/6.0 ; 
    mi[4][6] = 1.0/6.0 ; mi[4][7] =-1.0/4.0 ; mi[4][8] = 0.0     ;

    mi[5][0] = 1.0/9.0 ; mi[5][1] = 1.0/18.0; mi[5][2] = 1.0/36.0; 
    mi[5][3] = 1.0/6.0 ; mi[5][4] = 1.0/12.0; mi[5][5] = 1.0/6.0 ;
    mi[5][6] = 1.0/12.0; mi[5][7] = 0.0     ; mi[5][8] = 1.0/4.0 ;

    mi[6][0] = 1.0/9.0 ; mi[6][1] = 1.0/18.0; mi[6][2] = 1.0/36.0;
    mi[6][3] =-1.0/6.0 ; mi[6][4] =-1.0/12.0; mi[6][5] = 1.0/6.0 ;
    mi[6][6] = 1.0/12.0; mi[6][7] = 0.0     ; mi[6][8] =-1.0/4.0 ;

    mi[7][0] = 1.0/9.0 ; mi[7][1] = 1.0/18.0; mi[7][2] = 1.0/36.0;
    mi[7][3] =-1.0/6.0 ; mi[7][4] =-1.0/12.0; mi[7][5] =-1.0/6.0 ;
    mi[7][6] =-1.0/12.0; mi[7][7] = 0.0     ; mi[7][8] = 1.0/4.0 ;
  
    mi[8][0] = 1.0/9.0 ; mi[8][1] = 1.0/18.0; mi[8][2] = 1.0/36.0;
    mi[8][3] = 1.0/6.0 ; mi[8][4] = 1.0/12.0; mi[8][5] =-1.0/6.0 ;
    mi[8][6] =-1.0/12.0; mi[8][7] = 0.0     ; mi[8][8] =-1.0/4.0 ;

    for(i = 0; i <= 8; i++){ for(j = 0; j <= 8; j++){ sc[i][j] = 0.0; } }

    sc[0][0] = 0.0; sc[1][1] = 1.5    ; sc[2][2] = 1.4    ;
    sc[3][3] = 0.0; sc[4][4] = 1.5    ; sc[5][5] = 0.0    ;
    sc[6][6] = 1.5; sc[7][7] = 1.0/tau; sc[8][8] = 1.0/tau;
  }

  if(flag == 3){
// central moments
    mc[0][0] = 1.0; mc[0][1] = 1.0; mc[0][2] = 1.0;
    mc[0][3] = 1.0; mc[0][4] = 1.0; mc[0][5] = 1.0;
    mc[0][6] = 1.0; mc[0][7] = 1.0; mc[0][8] = 1.0;

    mc[1][0] = 0.0; mc[1][1] = 1.0; mc[1][2] = 0.0;
    mc[1][3] =-1.0; mc[1][4] = 0.0; mc[1][5] = 1.0;
    mc[1][6] =-1.0; mc[1][7] =-1.0; mc[1][8] = 1.0;

    mc[2][0] = 0.0; mc[2][1] = 0.0; mc[2][2] = 1.0;
    mc[2][3] = 0.0; mc[2][4] =-1.0; mc[2][5] = 1.0;
    mc[2][6] = 1.0; mc[2][7] =-1.0; mc[2][8] =-1.0;

    mc[3][0] = 0.0; mc[3][1] = 1.0; mc[3][2] = 1.0;
    mc[3][3] = 1.0; mc[3][4] = 1.0; mc[3][5] = 2.0;
    mc[3][6] = 2.0; mc[3][7] = 2.0; mc[3][8] = 2.0;

    mc[4][0] = 0.0; mc[4][1] = 1.0; mc[4][2] =-1.0;
    mc[4][3] = 1.0; mc[4][4] =-1.0; mc[4][5] = 0.0;
    mc[4][6] = 0.0; mc[4][7] = 0.0; mc[4][8] = 0.0;

    mc[5][0] = 0.0; mc[5][1] = 0.0; mc[5][2] = 0.0;
    mc[5][3] = 0.0; mc[5][4] = 0.0; mc[5][5] = 1.0;
    mc[5][6] =-1.0; mc[5][7] = 1.0; mc[5][8] =-1.0;

    mc[6][0] = 0.0; mc[6][1] = 0.0; mc[6][2] = 0.0;
    mc[6][3] = 0.0; mc[6][4] = 0.0; mc[6][5] = 1.0;
    mc[6][6] = 1.0; mc[6][7] =-1.0; mc[6][8] =-1.0;

    mc[7][0] = 0.0; mc[7][1] = 0.0; mc[7][2] = 0.0;
    mc[7][3] = 0.0; mc[7][4] = 0.0; mc[7][5] = 1.0;
    mc[7][6] =-1.0; mc[7][7] =-1.0; mc[7][8] = 1.0;

    mc[8][0] = 0.0; mc[8][1] = 0.0; mc[8][2] = 0.0;
    mc[8][3] = 0.0; mc[8][4] = 0.0; mc[8][5] = 1.0;
    mc[8][6] = 1.0; mc[8][7] = 1.0; mc[8][8] = 1.0;

// inverse matrix of the transformation matrix
    mi[0][0] = 1.00; mi[0][1] = 0.00; mi[0][2] = 0.00;
    mi[0][3] =-1.00; mi[0][4] = 0.00; mi[0][5] = 0.00;
    mi[0][6] = 0.00; mi[0][7] = 0.00; mi[0][8] = 1.00;

    mi[1][0] = 0.00; mi[1][1] = 0.50; mi[1][2] = 0.00;
    mi[1][3] = 0.25; mi[1][4] = 0.25; mi[1][5] = 0.00;
    mi[1][6] = 0.00; mi[1][7] =-0.50; mi[1][8] =-0.50;

    mi[2][0] = 0.00; mi[2][1] = 0.00; mi[2][2] = 0.50;
    mi[2][3] = 0.25; mi[2][4] =-0.25; mi[2][5] = 0.00;
    mi[2][6] =-0.50; mi[2][7] = 0.00; mi[2][8] =-0.50;

    mi[3][0] = 0.00; mi[3][1] =-0.50; mi[3][2] = 0.00;
    mi[3][3] = 0.25; mi[3][4] = 0.25; mi[3][5] = 0.00;
    mi[3][6] = 0.00; mi[3][7] = 0.50; mi[3][8] =-0.50;

    mi[4][0] = 0.00; mi[4][1] = 0.00; mi[4][2] =-0.50;
    mi[4][3] = 0.25; mi[4][4] =-0.25; mi[4][5] = 0.00;
    mi[4][6] = 0.50; mi[4][7] = 0.00; mi[4][8] =-0.50;

    mi[5][0] = 0.00; mi[5][1] = 0.00; mi[5][2] = 0.00;
    mi[5][3] = 0.00; mi[5][4] = 0.00; mi[5][5] = 0.25;
    mi[5][6] = 0.25; mi[5][7] = 0.25; mi[5][8] = 0.25;

    mi[6][0] = 0.00; mi[6][1] = 0.00; mi[6][2] = 0.00;
    mi[6][3] = 0.00; mi[6][4] = 0.00; mi[6][5] =-0.25;
    mi[6][6] = 0.25; mi[6][7] =-0.25; mi[6][8] = 0.25;

    mi[7][0] = 0.00; mi[7][1] = 0.00; mi[7][2] = 0.00;
    mi[7][3] = 0.00; mi[7][4] = 0.00; mi[7][5] = 0.25;
    mi[7][6] =-0.25; mi[7][7] =-0.25; mi[7][8] = 0.25;

    mi[8][0] = 0.00; mi[8][1] = 0.00; mi[8][2] = 0.00;
    mi[8][3] = 0.00; mi[8][4] = 0.00; mi[8][5] =-0.25;
    mi[8][6] =-0.25; mi[8][7] = 0.25; mi[8][8] = 0.25;

// relaxation matrix
    for(i = 0; i <= 8; i++){ for(j = 0; j <= 8; j++){ sc[i][j] = 0.0; } }

    sc[0][0] = 1.0; sc[1][1] = 1.0;     sc[2][2] = 1.0;
    sc[3][3] = 1.0; sc[4][4] = 1.0/tau; sc[5][5] = 1.0/tau;
    sc[6][6] = 1.0; sc[7][7] = 1.0;     sc[8][8] = 1.0;
  }

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
  for(loop1 = 0; loop1 < 100; loop1++){
  for(loop2 = 0; loop2 < 100; loop2++){
    time++;

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      un[i][j] = u[i][j]; vn[i][j] = v[i][j];
    } }

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

    if(flag == 1){
      for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
        f[k][i][j] = f[k][i][j] - (f[k][i][j] - f0[k][i][j])/tau;
      } } }
    } else if(flag == 2){
      for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
        t[k][i][j] = 0.0; t0[k][i][j] = 0.0; ftmp[k][i][j] = 0.0; fm[k][i][j] = 0.0;
      } } }

      for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
// transformation
        for(k = 0; k <= 8; k++){ for(m = 0; m <= 8; m++){
          t[k][i][j] = t[k][i][j] + mc[k][m] * f[m][i][j];
         t0[k][i][j] =t0[k][i][j] + mc[k][m] *f0[m][i][j];
        } }
// collision
        for(k = 0; k <= 8; k++){ for(m = 0; m <= 8; m++){
          ftmp[k][i][j] = ftmp[k][i][j] + sc[k][m] * (t[m][i][j] - t0[m][i][j]);
        } }
        for(k = 0; k <= 8; k++){
          t[k][i][j] = t[k][i][j] - ftmp[k][i][j];
        }
// inverse
        for(k = 0; k <= 8; k++){ for(m = 0; m <= 8; m++){
          fm[k][i][j] = fm[k][i][j] + mi[k][m] * t[m][i][j];
        } }

        for(k = 0; k <= 8; k++){
          f[k][i][j] = fm[k][i][j];
        }

      } }
    } else if(flag == 3){
      for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
// n: shift matrix
        nc[0][0] = 1.0;     nc[0][1] = 0.0; nc[0][2] = 0.0;
        nc[0][3] = 0.0;     nc[0][4] = 0.0; nc[0][5] = 0.0;
        nc[0][6] = 0.0;     nc[0][7] = 0.0; nc[0][8] = 0.0;

        nc[1][0] =-u[i][j]; nc[1][1] = 1.0; nc[1][2] = 0.0;
        nc[1][3] = 0.0;     nc[1][4] = 0.0; nc[1][5] = 0.0;
        nc[1][6] = 0.0;     nc[1][7] = 0.0; nc[1][8] = 0.0;

        nc[2][0] =-v[i][j]; nc[2][1] = 0.0; nc[2][2] = 1.0;
        nc[2][3] = 0.0;     nc[2][4] = 0.0; nc[2][5] = 0.0;
        nc[2][6] = 0.0;     nc[2][7] = 0.0; nc[2][8] = 0.0;

        nc[3][0] = pow(u[i][j],2) + pow(v[i][j],2);
        nc[3][1] =-2.0*u[i][j];
        nc[3][2] =-2.0*v[i][j];
        nc[3][3] = 1.0; nc[3][4] = 0.0; nc[3][5] = 0.0;
        nc[3][6] = 0.0; nc[3][7] = 0.0; nc[3][8] = 0.0;

        nc[4][0] = pow(u[i][j],2) - pow(v[i][j],2);
        nc[4][1] =-2.0*u[i][j];
        nc[4][2] = 2.0*v[i][j];
        nc[4][3] = 0.0; nc[4][4] = 1.0; nc[4][5] = 0.0;
        nc[4][6] = 0.0; nc[4][7] = 0.0; nc[4][8] = 0.0;

        nc[5][0] = u[i][j]*v[i][j];
        nc[5][1] =-v[i][j];
        nc[5][2] =-u[i][j];
        nc[5][3] = 0.0; nc[5][4] = 0.0; nc[5][5] = 1.0;
        nc[5][6] = 0.0; nc[5][7] = 0.0; nc[5][8] = 0.0;

        nc[6][0] =-pow(u[i][j],2) * v[i][j];
        nc[6][1] = 2.0*u[i][j]*v[i][j];
        nc[6][2] = pow(u[i][j],2);
        nc[6][3] =-v[i][j]*0.5;
        nc[6][4] =-v[i][j]*0.5;
        nc[6][5] =-2.0*u[i][j];
        nc[6][6] = 1.0; nc[6][7] = 0.0; nc[6][8] = 0.0;

        nc[7][0] =-pow(v[i][j],2) * u[i][j];
        nc[7][1] = pow(v[i][j],2);
        nc[7][2] = 2.0*u[i][j]*v[i][j];
        nc[7][3] =-u[i][j]*0.5;
        nc[7][4] = u[i][j]*0.5;
        nc[7][5] =-2.0*v[i][j];
        nc[7][6] = 0.0; nc[7][7] = 1.0; nc[7][8] = 0.0;

        nc[8][0] = pow(u[i][j],2) * pow(v[i][j],2);
        nc[8][1] =-2.0*u[i][j]*pow(v[i][j],2);
        nc[8][2] =-2.0*pow(u[i][j],2)*v[i][j];
        nc[8][3] = pow(v[i][j],2)*0.5 + pow(u[i][j],2)*0.5;
        nc[8][4] = pow(v[i][j],2)*0.5 - pow(u[i][j],2)*0.5;
        nc[8][5] = 4.0*u[i][j]*v[i][j];
        nc[8][6] =-2.0*v[i][j]; nc[8][7] =-2.0*u[i][j]; nc[8][8] = 1.0;

// inverse matrix of the shift matrix
        ni[0][0] = 1.0; ni[0][1] = 0.0; ni[0][2] = 0.0;
        ni[0][3] = 0.0; ni[0][4] = 0.0; ni[0][5] = 0.0;
        ni[0][6] = 0.0; ni[0][7] = 0.0; ni[0][8] = 0.0;

        ni[1][0] = u[i][j]; ni[1][1] = 1.0; ni[1][2] = 0.0;
        ni[1][3] = 0.0;     ni[1][4] = 0.0; ni[1][5] = 0.0;
        ni[1][6] = 0.0;     ni[1][7] = 0.0; ni[1][8] = 0.0;

        ni[2][0] = v[i][j]; ni[2][1] = 0.0; ni[2][2] = 1.0;
        ni[2][3] = 0.0;     ni[2][4] = 0.0; ni[2][5] = 0.0;
        ni[2][6] = 0.0;     ni[2][7] = 0.0; ni[2][8] = 0.0;

        ni[3][0] = pow(u[i][j],2) + pow(v[i][j],2);
        ni[3][1] = 2.0*u[i][j];
        ni[3][2] = 2.0*v[i][j];
        ni[3][3] = 1.0; ni[3][4] = 0.0; ni[3][5] = 0.0;
        ni[3][6] = 0.0; ni[3][7] = 0.0; ni[3][8] = 0.0;

        ni[4][0] = pow(u[i][j],2) - pow(v[i][j],2);
        ni[4][1] = 2.0*u[i][j];
        ni[4][2] =-2.0*v[i][j];
        ni[4][3] = 0.0; ni[4][4] = 1.0; ni[4][5] = 0.0;
        ni[4][6] = 0.0; ni[4][7] = 0.0; ni[4][8] = 0.0;

        ni[5][0] = u[i][j]*v[i][j]; ni[5][1] = v[i][j]; ni[5][2] = u[i][j];
        ni[5][3] = 0.0; ni[5][4] = 0.0; ni[5][5] = 1.0;
        ni[5][6] = 0.0; ni[5][7] = 0.0; ni[5][8] = 0.0;

        ni[6][0] = pow(u[i][j],2) * v[i][j];
        ni[6][1] = 2.0*u[i][j]*v[i][j];
        ni[6][2] = pow(u[i][j],2);
        ni[6][3] = v[i][j]*0.5;
        ni[6][4] = v[i][j]*0.5;
        ni[6][5] = 2.0*u[i][j];
        ni[6][6] = 1.0; ni[6][7] = 0.0; ni[6][8] = 0.0;

        ni[7][0] = pow(v[i][j],2) * u[i][j];
        ni[7][1] = pow(v[i][j],2);
        ni[7][2] = 2.0*u[i][j]*v[i][j];
        ni[7][3] = u[i][j]*0.5;
        ni[7][4] =-u[i][j]*0.5;
        ni[7][5] = 2.0*v[i][j];
        ni[7][6] = 0.0; ni[7][7] = 1.0; ni[7][8] = 0.0;

        ni[8][0] = pow(u[i][j],2) * pow(v[i][j],2);
        ni[8][1] = 2.0*u[i][j]*pow(v[i][j],2);
        ni[8][2] = 2.0*pow(u[i][j],2)*v[i][j];
        ni[8][3] = pow(v[i][j],2)*0.5 + pow(u[i][j],2)*0.5;
        ni[8][4] = pow(v[i][j],2)*0.5 - pow(u[i][j],2)*0.5;
        ni[8][5] = 4.0*u[i][j]*v[i][j];
        ni[8][6] = 2.0*v[i][j]; ni[8][7] = 2.0*u[i][j]; ni[8][8] = 1.0;

        for(k = 0; k <= 8; k++){
             t[k][i][j] = 0.0;  tn[k][i][j] = 0.0;
            t0[k][i][j] = 0.0; tn0[k][i][j] = 0.0;
          ftmp[k][i][j] = 0.0;  fm[k][i][j] = 0.0;
        }
// transformation
        for(k = 0; k <= 8; k++){ for(m = 0; m <= 8; m++){
          t[k][i][j] = t[k][i][j] + mc[k][m] * f[m][i][j];
         t0[k][i][j] =t0[k][i][j] + mc[k][m] *f0[m][i][j];
        } }
// shift
        for(k = 0; k <= 8; k++){ for(m = 0; m <= 8; m++){
          tn[k][i][j] = tn[k][i][j] + nc[k][m] * t[m][i][j];
         tn0[k][i][j] =tn0[k][i][j] + nc[k][m] *t0[m][i][j];
        } }

// collision
        for(k = 0; k <= 8; k++){ for(m = 0; m <= 8; m++){
          ftmp[k][i][j] = ftmp[k][i][j] + sc[k][m] * (tn[m][i][j] - tn0[m][i][j]);
        } }
        for(k = 0; k <= 8; k++){
          tn[k][i][j] = tn[k][i][j] - ftmp[k][i][j];
        }
// inverse
        for(k = 0; k <= 8; k++){
          ftmp[k][i][j] = 0.0;
        }
        for(k = 0; k <= 8; k++){ for(m = 0; m <= 8; m++){
          ftmp[k][i][j] = ftmp[k][i][j] + ni[k][m] * tn[m][i][j];
        } }

        for(k = 0; k <= 8; k++){ for(m = 0; m <= 8; m++){
          fm[k][i][j] = fm[k][i][j] + mi[k][m] * ftmp[m][i][j];
        } }
        for(k = 0; k <= 8; k++){
          f[k][i][j] =  fm[k][i][j];
        }
      } }
    }

    // propagation 
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
      ftmp[k][i][j] = f[k][i][j];
    } } }

    for(j = 0; j <= ny; j++){
      for(i = 0; i <= nx-1; i++){ f[1][i+1][j  ] = ftmp[1][i][j]; }
      for(i = 1; i <= nx  ; i++){ f[3][i-1][j  ] = ftmp[3][i][j]; }
    }

    for(j = 0; j <= ny-1; j++){
      for(i = 0; i <= nx  ; i++){ f[2][i  ][j+1] = ftmp[2][i][j]; }
      for(i = 0; i <= nx-1; i++){ f[5][i+1][j+1] = ftmp[5][i][j]; }
      for(i = 1; i <= nx  ; i++){ f[6][i-1][j+1] = ftmp[6][i][j]; }
    }
    for(j = 1; j <= ny; j++){
      for(i = 0; i <= nx  ; i++){ f[4][i  ][j-1] = ftmp[4][i][j]; }
      for(i = 1; i <= nx  ; i++){ f[7][i-1][j-1] = ftmp[7][i][j]; }
      for(i = 0; i <= nx-1; i++){ f[8][i+1][j-1] = ftmp[8][i][j]; }
    }
    // boundary condition
    // Half-way bounce back
    for(j = 1; j < ny; j++){
      jm = j - 1;
      jp = j + 1;
      f[1][   1][j] = f[3][0][j ];
      f[5][   1][j] = f[7][0][jm];
      f[8][   1][j] = f[6][0][jp];

      f[3][nx-1][j] = f[1][nx][j ];
      f[7][nx-1][j] = f[5][nx][jp];
      f[6][nx-1][j] = f[8][nx][jm];
    }

    for(i = 1; i < nx; i++){
      im = i - 1;
      ip = i + 1;
      f[2][i][   1] = f[4][i ][ 0];
      f[5][i][   1] = f[7][im][ 0];
      f[6][i][   1] = f[8][ip][ 0];

      rho[i][ny] = f[0][i][ny] + f[1][i][ny] + f[3][i][ny]
            + 2.0*(f[2][i][ny] + f[5][i][ny] + f[6][i][ny]);
      f[4][i][ny-1] = f[2][i ][ny] - rho[i][ny]*cx[2]*u0*2.0/3.0;
      f[7][i][ny-1] = f[5][ip][ny] - rho[i][ny]*cx[5]*u0/6.0;
      f[8][i][ny-1] = f[6][im][ny] - rho[i][ny]*cx[6]*u0/6.0;
    }
    // corner (rho = 1.0)
    u2 = u0*u0;      
    tmp = cx[6]*u0;      
    f[6][nx-1][ny-1] = (1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
    tmp = cx[8]*u0;
    f[8][nx-1][ny-1] = (1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;

    tmp = cx[5]*u0;
    f[5][   1][ny-1] = (1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
    tmp = cx[7]*u0;
    f[7][   1][ny-1] = (1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;

    f[6][nx-1][   1] = 1.0/36.0; f[8][nx-1][   1] = 1.0/36.0;
    f[5][   1][   1] = 1.0/36.0; f[7][   1][   1] = 1.0/36.0;

    for(i = 0; i <= nx; i++){
      u2 = u0*u0;
      f0[0][i][j] = (1.0 - 3.0/2.0*u2)*4.0/9.0;
      for(k = 1; k <= 4; k++){
        tmp = cx[k]*u0;
        f[k][i][ny] = (1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/6.0;
      }
      for(k = 5; k <= 8; k++){
        tmp = cx[k]*u0;
        f[k][i][ny] = (1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
      }
      f0[0][i][0] = 4.0/9.0;
      for(k = 1; k <= 4; k++){ f[k][i][0] = 1.0/6.0;  }
      for(k = 5; k <= 8; k++){ f[k][i][0] = 1.0/36.0; }
    }
    for(j = 0; j <= ny; j++){
      f0[0][0][j] = 4.0/9.0;
      for(k = 1; k <= 4; k++){ f[k][0][j] = 1.0/6.0;  }
      for(k = 5; k <= 8; k++){ f[k][0][j] = 1.0/36.0; }
      f0[0][nx][j] = 4.0/9.0;
      for(k = 1; k <= 4; k++){ f[k][nx][j] = 1.0/6.0;  }
      for(k = 5; k <= 8; k++){ f[k][nx][j] = 1.0/36.0; }
    }

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

    norm = 0.0;
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      tmp = sqrt(pow(u[i][j]-un[i][j],2) + pow(v[i][j]-vn[i][j],2));
      if(tmp > norm){ norm = tmp; }
    } }

  } //loop2
  
  if(flag == 1){ printf("SRT colision\n"); };
  if(flag == 2){ printf("MRT colision\n"); };
  if(flag == 3){ printf("central moment\n"); }; 
  printf("Re = %4.2f, u0= %5.3f, tau = %10.8f\n", re, u0, tau);
  printf("Time = %d, Norm = %8.6e\n", time, norm);

// stream function
  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
   sf[i][j] = 0.0;
  } }

  for(i = 0; i <= nx; i++){ for(j = 2; j <= ny; j++){
    sf[i][j] = (u[i][j] + 4.*u[i][j-1] + u[i][j-2])/3.0 + sf[i][j-2];
  } }
  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    sf[i][j] = sf[i][j]/u0/(double)(nx-1);
  } }

  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    a[i][j]='0';
  } }

  smax = -999.9; smin = 999.9;
  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = sf[i][j];
    if(tmp >= smax){ smax = tmp; }
    if(tmp <= smin){ smin = tmp; }
  } }

  printf("stream function: %10.8f, %10.8f\n", smax, smin);

  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = sf[i][j];
    if(tmp <=  smax*1.0            ){a[i][j] = '9';}
    if(tmp <= (smax*0.9 + smin*0.1)){a[i][j] = '8';}
    if(tmp <= (smax*0.8 + smin*0.2)){a[i][j] = '7';}
    if(tmp <= (smax*0.7 + smin*0.3)){a[i][j] = '6';}
    if(tmp <= (smax*0.6 + smin*0.4)){a[i][j] = '5';}
    if(tmp <= (smax*0.5 + smin*0.5)){a[i][j] = '4';}
    if(tmp <= (smax*0.4 + smin*0.6)){a[i][j] = '3';}
    if(tmp <= (smax*0.3 + smin*0.7)){a[i][j] = '2';}
    if(tmp <= (smax*0.2 + smin*0.8)){a[i][j] = '1';}
    if(tmp <= (smax*0.1 + smin*0.9)){a[i][j] = '0';}
  } }

  for(j = ny - 1; j >= 1; j = j - 2){ for(i = 1; i <= nx - 1; i = i + 2){
    printf("%c",a[i][j]);
    }
    printf("\n");
  }

  if(norm < 0.000000000001 && time > 10000){ exit(0); }

  } //loop1

  fp = fopen("dataCMu","w");
    for(j = 1; j <= ny-1; j++){ for(i = 1; i <= nx-1; i++){
      fprintf(fp," %10.8e", u[i][j])/u0;
    }
  fprintf(fp,"\n");
  } 
  fclose(fp);

  fp = fopen("dataCMv","w");
    for(j = 1; j <= ny-1; j++){ for(i = 1; i <= nx-1; i++){
      fprintf(fp," %10.8e", v[i][j]/u0);
    }
  fprintf(fp,"\n");
  } 
  fclose(fp);

  fp = fopen("dataCMs","w");
    for(j = 1; j <= ny-1; j++){ for(i = 1; i <= nx-1; i++){
      fprintf(fp," %10.8e", sf[i][j]);
    }
  fprintf(fp,"\n");
  } 
  fclose(fp);

  return 0;
}
