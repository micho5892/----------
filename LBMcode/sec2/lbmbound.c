// Poiseuille flow (scheme for the boundary condition)
// Boundary Condition for Lattice Boltzmann Method.
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
// ut : horizontal velocity component of the top wall
// vt : vertical velocity component of the top wall
// ub : horizontal velocity component of the bottom wall
// vb : vertical velocity component of the bottom wall
// ue : analytical solution
// gx : body force (x-axis)
// gy : body force (y-axis)
// h  : channel width
// nx : number of grid points (x-axis)
// ny : number of grid points (y-axis)
// f  : distribution function
// f0 : equilibrium distribution function
// cx : discrete velocity (x-axis)
// cy : discrete velocity (y-axis)
// tau: relaxation time
// nu : kinematic viscosity
// q  : boundary position
// us : velocity slip
// stress: stress on the boundary
// err: relative error
// flag = 1 (Equilibrium)
// flag = 2 (On-grid bounce back)
// flag = 3 (No-slip boundary (Inamuro))
// flag = 4 (Non-equilibrium bounce back (Zou))
// flag = 5 (Half-way bounce back)
// flag = 6 (Interpolated bounce back (Linear))
// flag = 7 (Interpolated bounce back (Quadratic))

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

# define DIM 24

int main(void)
{
  FILE   *fp;
  int    nx = 20, ny = 20, time = 0, loop1, loop2;
  int    i, j, k, in, jn, ip, im, nb, flag;
  double u[DIM][DIM], v[DIM][DIM], un[DIM][DIM], vn[DIM][DIM], ue[DIM][DIM], rho[DIM][DIM];
  double f[9][DIM][DIM], f0[9][DIM][DIM], ftmp[9][DIM][DIM], cx[9], cy[9];
  double gx = 0.00001, gy = 0.0, ut = 0.0, vt = 0.0, ub = 0.0, vb = 0.0, q = 0.25;
  double h, tmp, u2, nu, norm, err, rhod, us, stress;
  double tau = 0.56;
//  double tau = 3.00;
  char   a[DIM][DIM];

  flag = 1;
  printf("Select number (1- 7)\n");
  printf("1: Equilibrium\n");
  printf("2: On-grid bounce back\n");
  printf("3: No-slip boundary (Inamuro)\n");
  printf("4: Non-equilibrium bounce back (Zou)\n");
  printf("5: Half-way bounce back\n");
  printf("6: Interpolated bounce back (Linear)\n");
  printf("7: Interpolated bounce back (Quadratic)\n");
  printf("Enter  1- 7 \n");
  scanf("%d", &flag);

  // initial condition
  nu = (tau - 0.5)/3.0;

  h = (double)ny;

  if(flag == 5){ny = (int)h + 1;}
  if(flag >= 6){ny = (int)h + 1; h = (double)ny - 2.0 + 2.0*q;}

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    u[i][j] = 0.0; v[i][j] = 0.0; un[i][j] = 0.0; vn[i][j] = 0.0; rho[i][j] = 1.0;

    ue[i][j] =  (double)j/h*ut + ub + gx/2.0/nu*(h - (double)j)*(double)j;
    if(flag == 5){
      ue[i][j] = ((double)j - 0.5    )/h * ut + ub
               + gx/2.0/nu*(h + 0.5 - (double)j)*((double)j - 0.5); }

    if(flag >= 6){
       ue[i][j] = ((double)j - 1.0 + q)/h * ut + ub
                + gx/2.0/nu*(h + 1.0 - q - (double)j)*((double)j - 1.0 + q);}
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
  for(loop1 = 0; loop1 < 500; loop1++){
  for(loop2 = 0; loop2 < 200; loop2++){
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

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
      f[k][i][j] = f[k][i][j] - (f[k][i][j] - f0[k][i][j])/tau;
    } } }

    // force
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      for(k = 1; k <= 4; k++){
        f[k][i][j] = f[k][i][j] + rho[i][j]*(cx[k]*gx + cy[k]*gy)/3.0;
      }
      for(k = 5; k <= 8; k++){
        f[k][i][j] = f[k][i][j] + rho[i][j]*(cx[k]*gx + cy[k]*gy)/12.0;
      }
    } }

    // propagation 
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
      ftmp[k][i][j] = f[k][i][j];
    } } }

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
      in = i + (int)cx[k]; jn = j + (int)cy[k];
      if(in == nx + 1){in =  0;} if(in == -1){in = nx;}
      if(jn == ny + 1){jn =  0;} if(jn == -1){jn = ny;}
      f[k][in][jn] = ftmp[k][i][j];
    } } }

    // boundary condition
    if(flag == 1){
    // Equilibrium
      for(i = 0; i <= nx; i++){
        rho[i][0] = (f[0][i][0] + f[1][i][0] + f[3][i][0]
              + 2.0*(f[4][i][0] + f[7][i][0] + f[8][i][0]))/(1.0 - vb);
        u2 = ub*ub + vb*vb;
        f[0][i][ 0] = rho[i][ 0]*(1.0 -3.0/2.0*u2)*4.0/9.0;
        for(k = 1; k <= 4; k++){
          tmp = cx[k]*ub + cy[k]*vb;      
          f[k][i][ 0] = rho[i][ 0]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/9.0;
        }  
        for(k = 5; k <= 8; k++){
          tmp = cx[k]*ub + cy[k]*vb;      
          f[k][i][ 0] = rho[i][ 0]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
        }

        rho[i][ny] = (f[0][i][ny] + f[1][i][ny] + f[3][i][ny]
               + 2.0*(f[2][i][ny] + f[5][i][ny] + f[6][i][ny]))/(1.0 + vt);
        u2 = ut*ut + vt*vt;
        f[0][i][ny] = rho[i][ny]*(1.0 -3.0/2.0*u2)*4.0/9.0;
        for(k = 1; k <= 4; k++){
          tmp = cx[k]*ut + cy[k]*vt;      
          f[k][i][ny] = rho[i][ny]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/9.0;
        }  
        for(k = 5; k <= 8; k++){
          tmp = cx[k]*ut + cy[k]*vt;      
          f[k][i][ny] = rho[i][ny]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
        }
      } 
    }else if(flag == 2){
    // On-grid bounce back
      for(i = 0; i <= nx; i++){
        rho[i][0] = (f[0][i][0] + f[1][i][0] + f[3][i][0]
              + 2.0*(f[4][i][0] + f[7][i][0] + f[8][i][0]))/(1.0 - vb);
        f[2][i][0] = f[4][i][0] - rho[i][0]*(cx[4]*ub + cy[4]*vb)*2.0/3.0;
        f[5][i][0] = f[7][i][0] - rho[i][0]*(cx[7]*ub + cy[7]*vb)/6.0;
        f[6][i][0] = f[8][i][0] - rho[i][0]*(cx[8]*ub + cy[8]*vb)/6.0;

        rho[i][ny] = (f[0][i][ny] + f[1][i][ny] + f[3][i][ny]
               + 2.0*(f[2][i][ny] + f[5][i][ny] + f[6][i][ny]))/(1.0 + vt);
        f[4][i][ny] = f[2][i][ny] - rho[i][ny]*(cx[2]*ut + cy[2]*vt)*2.0/3.0;
        f[7][i][ny] = f[5][i][ny] - rho[i][ny]*(cx[5]*ut + cy[5]*vt)/6.0;
        f[8][i][ny] = f[6][i][ny] - rho[i][ny]*(cx[6]*ut + cy[6]*vt)/6.0;
      } 
    }else if(flag == 3){
    // No-slip boundary (Inamuro)
      for(i = 0; i <= nx; i++){
        rhod = 6.0*( rho[i][0]*vb + f[4][i][0] + f[7][i][0] + f[8][i][0])
           / (1. + 3.0*vb + 3.0*vb*vb);
        us = 6.0*( rho[i][0]*ub - f[1][i][0] + f[3][i][0] + f[7][i][0] - f[8][i][0])
           /rhod/(1.0 +3.0*vb) - ub;

        u2 = (ub + us)*(ub + us) + vb*vb;
        tmp = cx[2]*(ub + us) + cy[2]*vb;
        f[2][i][ 0] = rhod*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/9.0;
        tmp = cx[5]*(ub + us) + cy[5]*vb;
        f[5][i][ 0] = rhod*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
        tmp = cx[6]*(ub + us) + cy[6]*vb;
        f[6][i][ 0] = rhod*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;

        rhod = 6.0*(-rho[i][ny]*vt + f[2][i][ny] + f[5][i][ny] + f[6][i][ny])
           / (1. - 3.0*vt + 3.0*vt*vt);
        us = 6.0*( rho[i][ny]*ut - f[1][i][ny] + f[3][i][ny] - f[5][i][ny] + f[6][i][ny])
           /rhod/(1.0 -3.0*vt) - ut;

        u2 = (ut + us)*(ut + us) + vt*vt;
        tmp = cx[4]*(ut + us) + cy[4]*vt;
        f[4][i][ny] = rhod*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/9.0;
        tmp = cx[7]*(ut + us) + cy[7]*vt;
        f[7][i][ny] = rhod*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
        tmp = cx[8]*(ut + us) + cy[8]*vt;
        f[8][i][ny] = rhod*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
      } 

    }else if(flag == 4){
    // Non-equilibrium bounce back (Zou)
      for(i = 0; i <= nx; i++){
        rho[i][0] = (f[0][i][0] + f[1][i][0] + f[3][i][0]
              + 2.0*(f[4][i][0] + f[7][i][0] + f[8][i][0]))/(1.0 - vb);
        f[2][i][0] = f[4][i][0] + rho[i][0]*vb*2.0/3.0;
        f[5][i][0] = f[7][i][0] - 0.5*(f[1][i][0] - f[3][i][0])
                   + rho[i][0]*( ub/2.0 + vb/6.0);
        f[6][i][0] = f[8][i][0] + 0.5*(f[1][i][0] - f[3][i][0])
                   + rho[i][0]*(-ub/2.0 + vb/6.0);

        rho[i][ny] = (f[0][i][ny] + f[1][i][ny] + f[3][i][ny]
               + 2.0*(f[2][i][ny] + f[5][i][ny] + f[6][i][ny]))/(1.0 + vt);
        f[4][i][ny] = f[2][i][ny] - rho[i][ny]*vt*2.0/3.0;
        f[7][i][ny] = f[5][i][ny] + 0.5*(f[1][i][ny] - f[3][i][ny])
                    - rho[i][ny]*( ut/2.0 + vt/6.0);
        f[8][i][ny] = f[6][i][ny] - 0.5*(f[1][i][ny] - f[3][i][ny])
                    - rho[i][ny]*(-ut/2.0 + vt/6.0);
      } 
    }else if(flag == 5){
    // Half-way bounce back
      for(i = 0; i <= nx; i++){
        im = i - 1; if(i ==  0){im = nx;}
        ip = i + 1; if(i == nx){ip =  0;}
        rho[i][0] = (f[0][i][0] + f[1][i][0] + f[3][i][0]
              + 2.0*(f[4][i][0] + f[7][i][0] + f[8][i][0]))/(1.0 - vb);
        f[2][i][   1] = f[4][i ][ 0] - rho[i][0]*(cx[4]*ub + cy[4]*vb)*2.0/3.0;
        f[5][i][   1] = f[7][im][ 0] - rho[i][0]*(cx[7]*ub + cy[7]*vb)/6.0;
        f[6][i][   1] = f[8][ip][ 0] - rho[i][0]*(cx[8]*ub + cy[8]*vb)/6.0;

        rho[i][ny] = (f[0][i][ny] + f[1][i][ny] + f[3][i][ny]
               + 2.0*(f[2][i][ny] + f[5][i][ny] + f[6][i][ny]))/(1.0 + vt);
        f[4][i][ny-1] = f[2][i ][ny] - rho[i][ny]*(cx[2]*ut + cy[2]*vt)*2.0/3.0;
        f[7][i][ny-1] = f[5][ip][ny] - rho[i][ny]*(cx[5]*ut + cy[5]*vt)/6.0;
        f[8][i][ny-1] = f[6][im][ny] - rho[i][ny]*(cx[6]*ut + cy[6]*vt)/6.0;

        stress = cx[4]*f[4][i][0] - cx[2]*f[2][i][1]
               + cx[7]*f[7][i][0] - cx[5]*f[5][i][1]
               + cx[8]*f[8][i][0] - cx[6]*f[6][i][1];
      } 
    }else if(flag == 6){
    // Interpolated bounce back (Linear)
      for(i = 0; i <= nx; i++){
        ip = i + 1; if(i == nx){ip =  0;}
        im = i - 1; if(i ==  0){im = nx;}
        rho[i][0] = (f[0][i][0] + f[1][i][0] + f[3][i][0]
              + 2.0*(f[4][i][0] + f[7][i][0] + f[8][i][0]))/(1.0 - vb);
        if(q <= 0.5){
          f[2][i][1] = f[4][i ][0]*2.0*q + f[4][i][1]*(1.0 - 2.0*q) - rho[i][0]*(cx[4]*ub + cy[4]*vb)*2.0/3.0;
          f[5][i][1] = f[7][im][0]*2.0*q + f[7][i][1]*(1.0 - 2.0*q) - rho[i][0]*(cx[7]*ub + cy[7]*vb)/6.0;
          f[6][i][1] = f[8][ip][0]*2.0*q + f[8][i][1]*(1.0 - 2.0*q) - rho[i][0]*(cx[8]*ub + cy[8]*vb)/6.0;
        }else{
          f[2][i][1] = f[4][i ][0]/2.0/q + f[2][i ][2]*(2.0*q - 1.0)/2.0/q - rho[i][0]*(cx[4]*ub + cy[4]*vb)/3.0/q;
          f[5][i][1] = f[7][im][0]/2.0/q + f[5][ip][2]*(2.0*q - 1.0)/2.0/q - rho[i][0]*(cx[7]*ub + cy[7]*vb)/12.0/q;
          f[6][i][1] = f[8][ip][0]/2.0/q + f[6][im][2]*(2.0*q - 1.0)/2.0/q - rho[i][0]*(cx[8]*ub + cy[8]*vb)/12.0/q;
        } 

        rho[i][ny] = (f[0][i][ny] + f[1][i][ny] + f[3][i][ny]
               + 2.0*(f[2][i][ny] + f[5][i][ny] + f[6][i][ny]))/(1.0 + vt);
        if(q <= 0.5){
          f[4][i][ny-1] = f[2][i ][ny]*2.0*q + f[2][i][ny-1]*(1.0 - 2.0*q) - rho[i][ny]*(cx[2]*ut + cy[2]*vt)*2.0/3.0;
          f[7][i][ny-1] = f[5][ip][ny]*2.0*q + f[5][i][ny-1]*(1.0 - 2.0*q) - rho[i][ny]*(cx[5]*ut + cy[5]*vt)/6.0;
          f[8][i][ny-1] = f[6][im][ny]*2.0*q + f[6][i][ny-1]*(1.0 - 2.0*q) - rho[i][ny]*(cx[6]*ut + cy[6]*vt)/6.0;
        }else{
          f[4][i][ny-1] = f[2][i ][ny]/2.0/q + f[4][i ][ny-2]*(2.0*q - 1.0)/2.0/q
                        - rho[i][ny]*(cx[2]*ut + cy[2]*vt)/3.0/q;
          f[7][i][ny-1] = f[5][ip][ny]/2.0/q + f[7][im][ny-2]*(2.0*q - 1.0)/2.0/q
                        - rho[i][ny]*(cx[5]*ut + cy[5]*vt)/12.0/q;
          f[8][i][ny-1] = f[6][im][ny]/2.0/q + f[8][ip][ny-2]*(2.0*q - 1.0)/2.0/q
                        - rho[i][ny]*(cx[6]*ut + cy[6]*vt)/12.0/q;
        } 
      } 
    }else if(flag == 7){
    // Interpolated bounce back (Quadratic)
      for(i = 0; i <= nx; i++){
        ip = i + 1; if(i == nx){ip =  0;}
        im = i - 1; if(i ==  0){im = nx;}

        rho[i][0] = (f[0][i][0] + f[1][i][0] + f[3][i][0]
              + 2.0*(f[4][i][0] + f[7][i][0] + f[8][i][0]))/(1.0 - vb);
        if(q <= 0.5){
          f[2][i][1] = f[4][i ][0]*(1.0 + 2.0*q)*q + f[4][i][1]*(1.0 + 2.0*q)*(1.0 - 2.0*q)
                     - f[4][i ][2]*(1.0 - 2.0*q)*q - rho[i][0]*(cx[4]*ub + cy[4]*vb)*2.0/3.0;
          f[5][i][1] = f[7][im][0]*(1.0 + 2.0*q)*q + f[7][i][1]*(1.0 + 2.0*q)*(1.0 - 2.0*q)
                     - f[7][ip][2]*(1.0 - 2.0*q)*q - rho[i][0]*(cx[7]*ub + cy[7]*vb)/6.0;
          f[6][i][1] = f[8][ip][0]*(1.0 + 2.0*q)*q + f[8][i][1]*(1.0 + 2.0*q)*(1.0 - 2.0*q)
                     - f[8][im][2]*(1.0 - 2.0*q)*q - rho[i][0]*(cx[8]*ub + cy[8]*vb)/6.0;
        }else{
          f[2][i][1] = f[4][i ][0]/(1.0 + 2.0*q)/q + f[2][i][2]*(2.0*q - 1.0)/q 
                     - f[2][i ][3]*(2.0*q - 1.0)/(2.0*q + 1.0)
                     - rho[i][0]*(cx[4]*ub + cy[4]*vb)*2.0/3.0/(2.0*q + 1.0)/q;
          in = ip + 1; if(ip == nx){in = 0;}
          f[5][i][1] = f[7][i ][0]/(1.0 + 2.0*q)/q + f[5][ip][2]*(2.0*q - 1.0)/q 
                     - f[5][in][3]*(2.0*q - 1.0)/(2.0*q + 1.0)
                     - rho[i][0]*(cx[7]*ub + cy[7]*vb)/6.0/(2.0*q + 1.0)/q;
          in = im - 1; if(im==  0){in = nx;}
          f[6][i][1] = f[8][ip][0]/(1.0 + 2.0*q)/q + f[6][im][2]*(2.0*q - 1.0)/q 
                     - f[6][in][3]*(2.0*q - 1.0)/(2.0*q + 1.0)
                     - rho[i][0]*(cx[8]*ub + cy[8]*vb)/6.0/(2.0*q + 1.0)/q;
        } 

        rho[i][ny] = (f[0][i][ny] + f[1][i][ny] + f[3][i][ny]
               + 2.0*(f[2][i][ny] + f[5][i][ny] + f[6][i][ny]))/(1.0 + vt);
        if(q <= 0.5){
          f[4][i][ny-1] = f[2][i ][ny]*(1.0 + 2.0*q)*q + f[2][i][ny-1]*(1.0 + 2.0*q)*(1.0 - 2.0*q)
                        - f[2][i ][ny-2]*(1.0 - 2.0*q)*q - rho[i][ny]*(cx[2]*ut + cy[2]*vt)*2.0/3.0;
          f[7][i][ny-1] = f[5][ip][ny]*(1.0 + 2.0*q)*q + f[5][i][ny-1]*(1.0 + 2.0*q)*(1.0 - 2.0*q)
                        - f[5][im][ny-2]*(1.0 - 2.0*q)*q - rho[i][ny]*(cx[5]*ut + cy[5]*vt)/6.0;
          f[8][i][ny-1] = f[6][im][ny]*(1.0 + 2.0*q)*q + f[6][i][ny-1]*(1.0 + 2.0*q)*(1.0 - 2.0*q)
                        - f[6][ip][ny-2]*(1.0 - 2.0*q)*q - rho[i][ny]*(cx[6]*ut + cy[6]*vt)/6.0;
        }else{
          f[4][i][ny-1] = f[2][i ][ny]/(1.0 + 2.0*q)/q + f[4][i][ny-2]*(2.0*q - 1.0)/q 
                        - f[4][i ][ny-3]*(2.0*q - 1.0)/(2.0*q + 1.0)
                        - rho[i][ny]*(cx[2]*ut + cy[2]*vt)*2.0/3.0/(2.0*q + 1.0)/q;
          in = im - 1; if(im==  0){in = nx;}
          f[7][i][ny-1] = f[5][i ][ny]/(1.0 + 2.0*q)/q + f[7][im][ny-2]*(2.0*q - 1.0)/q 
                        - f[7][in][ny-3]*(2.0*q - 1.0)/(2.0*q + 1.0)
                        - rho[i][ny]*(cx[5]*ut + cy[5]*vt)/6.0/(2.0*q + 1.0)/q;
          in = ip + 1; if(ip == nx){in = 0;}
          f[8][i][ny-1] = f[6][ip][ny]/(1.0 + 2.0*q)/q + f[8][ip][ny-2]*(2.0*q - 1.0)/q 
                        - f[8][in][ny-3]*(2.0*q - 1.0)/(2.0*q + 1.0)
                        - rho[i][ny]*(cx[6]*ut + cy[6]*vt)/6.0/(2.0*q + 1.0)/q;
        } 
      } 
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

    err = 0.0; tmp = 0.0;
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      err = err + pow(ue[i][j] - u[i][j],2);
      tmp = tmp + pow(ue[i][j],2);
    } }
    err = sqrt(err/tmp);

  } //loop2
  
  printf("Time = %d, Norm = %8.6e\n", time, norm);
  if(flag == 1){ printf("Error = %8.6e(Equilibrium)\n",err);
                 if(gx > 0.0){ printf("Slip = %8.6e(%6.4e)\n",
                   u[nx/2][0]*8.0*nu/gx/h/h,8.0*tau*(2.0*tau - 1.0)/3.0/h/h);}
                 printf("ub[0] = %8.6e(%6.4e), ut[0] = %8.6e(%6.4e)\n",u[nx/2][0],ub,u[nx/2][ny],ut);
                 printf("ub[1] = %8.6e(%6.4e), ut[1] = %8.6e(%6.4e)\n",
                 u[nx/2][1],ue[nx/2][1],u[nx/2][ny-1],ue[nx/2][ny-1]);
               }
  if(flag == 2){ printf("Error = %8.6e(On-grid bounce back)\n",err);
                 if(gx > 0.0){ printf("Slip = %8.6e(%6.4e)\n",
                   u[nx/2][0]*8.0*nu/gx/h/h,8.0*tau*(2.0*tau - 1.0)/3.0/h/h);}
                 printf("ub[0] = %8.6e(%6.4e), ut[0] = %8.6e(%6.4e)\n",u[nx/2][0],ub,u[nx/2][ny],ut);
                 printf("ub[1] = %8.6e(%6.4e), ut[1] = %8.6e(%6.4e)\n",
                 u[nx/2][1],ue[nx/2][1],u[nx/2][ny-1],ue[nx/2][ny-1]);
                 fp = fopen("slipbb.txt","w");
                 fprintf(fp,"%16.8e\n", u[nx/2][0]*8.0*nu/gx/h/h);
                 fclose(fp);
               }
  if(flag == 3){ printf("Error = %8.6e(No-slip boundary (Inamuro))\n",err);
                 printf("ub[0] = %8.6e(%6.4e), ut[0] = %8.6e(%6.4e)\n",u[nx/2][0],ub,u[nx/2][ny],ut);
                 printf("ub[1] = %8.6e(%6.4e), ut[1] = %8.6e(%6.4e)\n",
                 u[nx/2][1],ue[nx/2][1],u[nx/2][ny-1],ue[nx/2][ny-1]);
               }
  if(flag == 4){ printf("Error = %8.6e(Non-equilibrium bounce back (Zou))\n",err);
                 printf("ub[0] = %8.6e(%6.4e), ut[0] = %8.6e(%6.4e)\n",u[nx/2][0],ub,u[nx/2][ny],ut);
                 printf("ub[1] = %8.6e(%6.4e), ut[1] = %8.6e(%6.4e)\n",
                 u[nx/2][1],ue[nx/2][1],u[nx/2][ny-1],ue[nx/2][ny-1]);
               }
  if(flag == 5){ printf("Error = %8.6e(Half-way bounce back)\n",err);
                 if(gx > 0.0){
                   printf("Slip = %8.6e(%6.4e)\n",
                   (u[nx/2][1] - ue[nx/2][1])*8.0*nu/gx/h/h,(16.0*tau*tau - 20.0*tau + 3.0)/3.0/h/h);
                   printf("Stress = %8.6e(%6.4e)\n", stress, (gx/8.0/nu*h*h)*4.0*nu/h);
                 }
                 printf("ub[1] = %8.6e(%6.4e), ut[1] = %8.6e(%6.4e)\n",
                 u[nx/2][1],ue[nx/2][1],u[nx/2][ny-1],ue[nx/2][ny-1]);
                 printf("ub[2] = %8.6e(%6.4e), ut[2] = %8.6e(%6.4e)\n",
                 u[nx/2][2],ue[nx/2][2],u[nx/2][ny-2],ue[nx/2][ny-2]);

                 fp = fopen("sliphbb.txt","w");
                 fprintf(fp,"%16.8e\n",(u[nx/2][1] - ue[nx/2][1])*8.0*nu/gx/h/h);
                 fclose(fp);
               }
  if(flag == 6){ printf("Error = %8.6e(Interpolated bounce back (Linear))\n",err);
                 if(gx > 0.0 || q <0.5){ printf("Slip = %8.6e(%6.4e)\n",
                   (u[nx/2][1] - ue[nx/2][1])*8.0*nu/gx/h/h,(16.0*tau*tau - 8.0*tau - 24.0*q*tau +12.0*q - 12.0*q*q)/3.0/h/h);}
                 printf("ub[1] = %8.6e(%6.4e), ut[1] = %8.6e(%6.4e)\n",
                 u[nx/2][1],ue[nx/2][1],u[nx/2][ny-1],ue[nx/2][ny-1]);
                 printf("ub[2] = %8.6e(%6.4e), ut[2] = %8.6e(%6.4e)\n",
                 u[nx/2][2],ue[nx/2][2],u[nx/2][ny-2],ue[nx/2][ny-2]);

                 fp = fopen("slipibb.txt","w");
                 fprintf(fp,"%16.8e\n",(u[nx/2][1] - ue[nx/2][1])*8.0*nu/gx/h/h);
                 fclose(fp);
               }
  if(flag == 7){ printf("Error = %8.6e(Interpolated bounce back (Quadratic))\n",err);
                 if(gx > 0.0 || q <0.5){ printf("Slip = %8.6e(%6.4e)\n",
                   (u[nx/2][1] - ue[nx/2][1])*8.0*nu/gx/h/h,(16.0*tau*tau - 8.0*tau - 24.0*q*tau + 12.0*q*q)/3.0/h/h);}
                 printf("ub[1] = %8.6e(%6.4e), ut[1] = %8.6e(%6.4e)\n",
                 u[nx/2][1],ue[nx/2][1],u[nx/2][ny-1],ue[nx/2][ny-1]);
                 printf("ub[2] = %8.6e(%6.4e), ut[2] = %8.6e(%6.4e)\n",
                 u[nx/2][2],ue[nx/2][2],u[nx/2][ny-2],ue[nx/2][ny-2]);
                }


  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    a[i][j]='0';
  } }

  for(j = 0; j <= ny; j++){
    nb = u[nx/2][j]/(ut + gx/nu/8.0*h*h)*nx;
    for(i = 0; i <= nb; i++){
      a[i][j]= '-';
    }
  }

  if(flag <= 4){
    for(j = ny; j >= 0; j--){ for(i = 0; i <= nx; i++){
      printf("%c",a[i][j]);
      }
      printf("\n");
    }
  }else {
    for(j = ny - 1; j >= 1; j--){ for(i = 0; i <= nx; i++){
      printf("%c",a[i][j]);
      }
      printf("\n");
    }
  }

  if(norm < 0.0000000001 && time > 10000){ 

    fp = fopen("data","w");
    for(j = 0; j <= ny; j++){
     fprintf(fp,"%10.8e\n", u[nx/2][j]/(gx/nu/8.0*h*h));
    }
    fclose(fp);

    if(flag >= 5){
     fp = fopen("data","w");
      for(j = 1; j <= ny-1; j++){
       fprintf(fp,"%10.8e\n", u[nx/2][j]/(gx/nu/8.0*h*h));
      }
      fclose(fp);
    }

    fp = fopen("error","w");
    fprintf(fp,"%15.8e\n", err);
    fclose(fp);

    exit(0);
  }

  } //loop1

  return 0;
}
