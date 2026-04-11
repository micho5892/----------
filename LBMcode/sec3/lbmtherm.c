// Thermal Lattice Boltzmann Method.
// The source code is written in C programming language.
// Copyright @2021 Takeshi Seta All Rights Reserved.
//
//    2
//    |
// 3--0--1
//    |
//    4
//
// e   : temperature (numerical solution, time = n    )
// en  : temperature (numerical solution, time = n - 1)
// ea  : temperature (analitycal solution)
// u   : horizontal velocity component (given)
// v   : vertical velocity component (given)
// u0  : characteristic velocity
// gx  : body force (x-axis)
// gy  : body force (y-axis)
// nx  : number of grid points (x-axis)
// ny  : number of grid points (y-axis)
// g   : distribution function
// g0  : equilibrium distribution function
// cx  : discrete velocity (x-axis)
// cy  : discrete velocity (y-axis)
// taug: relaxation time
// chi : diffusion coefficient
// wav : wavenumber
// h   : channel width
// pe  : Peclet number
// q   : boundary position
// err : relative error
// wav : wavenumber
// pe  : Peclet number
// M_PI: pi
// I   : imaginary unit
// flag = 1 (Dirichlet boundary codition)
// flag = 2 (Neumann   boundary codition)

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <complex.h>

# define DIM 100

int main(void)
{
  FILE   *fp;
  int    nx = 64, ny = 64, loop1, loop2;
  int    i, j, k, in, jn, ip, im, time = 0, flag;
  double e[DIM][DIM], en[DIM][DIM], ea[DIM][DIM], u[DIM][DIM], v[DIM][DIM];
  double g[5][DIM][DIM], g0[5][DIM][DIM], gtmp[5][DIM][DIM], cx[5], cy[5];
  double h, tmp, norm, err, emax, emin, u0, chi, wav;
  double pe = 20.0, taug = 0.56, q = 0.7;
  double _Complex beta, ec[DIM][DIM];
  char   a[DIM][DIM], b[DIM][DIM];

  // initial condition
  wav = 2.*M_PI/(double)nx;
  chi = (taug - 0.5)/3.0;
  h = (double)(ny - 2) + 2.0*q; u0 = pe*chi/h;
  beta = wav*csqrt(1.0 + I*u0/chi/wav);

  cx[0] = 0.0; cy[0] = 0.0;
  cx[1] = 1.0; cy[1] = 0.0;  cx[2] = 0.0; cy[2] = 1.0;
  cx[3] =-1.0; cy[3] = 0.0;  cx[4] = 0.0; cy[4] =-1.0;

  flag = 1; // Dirichlet boundary codition
//  flag = 2; // Neumann   boundary codition

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    u[i][j] = u0; v[i][j] = 0.0; e[i][j] = 1.0;
  } }

  if(flag == 1){
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      ec[i][j] = cexp(I*wav*(double)i)/(cexp(beta*h) - cexp(-beta*h))
               *( (1.0 - cexp(-beta*h))*cexp( beta*((double)j - 1.0 + q))
                - (1.0 - cexp( beta*h))*cexp(-beta*((double)j - 1.0 + q)));
    } }
  }else{
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      ec[i][j] = cexp(I*wav*(double)i)/beta/h/(cexp(beta*h) - cexp(-beta*h))
               *( (1.0 + cexp(-beta*h))*cexp( beta*((double)j - 1.0 + q))
                + (1.0 + cexp( beta*h))*cexp(-beta*((double)j - 1.0 + q)));
    } }
  }

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    ea[i][j] = creal(ec[i][j]);
     e[i][j] = ea[i][j];
  } }

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    g0[0][i][j] = e[i][j]/3.0;
    for(k = 1; k <= 4; k++){
      tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
      g0[k][i][j] = e[i][j]*(1.0 + 3.0*tmp)/6.0;
    }
  } }

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 4; k++){
    g[k][i][j] = g0[k][i][j];
  } } }

  for(loop1 = 0; loop1 < 100; loop1++){
  for(loop2 = 0; loop2 < 500; loop2++){
    time++;

    norm = 0.0;
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      tmp = fabs(e[i][j] - en[i][j]);
      if(tmp > norm){ norm = tmp; }
    } }

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ en[i][j] = e[i][j]; } }

    // collision
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      g0[0][i][j] = e[i][j]/3.0;
      for(k = 1; k <= 4; k++){
        tmp = cx[k]*u[i][j] + cy[k]*v[i][j]; 
        g0[k][i][j] = e[i][j]*(1.0 + 3.0*tmp)/6.0;
      }
    } }

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 4; k++){
      g[k][i][j] = g[k][i][j] - (g[k][i][j] - g0[k][i][j])/taug;
    } } }

    // propagation 
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 4; k++){
      gtmp[k][i][j] = g[k][i][j];
    } } }

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 4; k++){
      in = i + (int)cx[k]; jn = j + (int)cy[k];
      if(in == nx + 1){in =  0;} if(in == -1){in = nx;}
      if(jn == ny + 1){jn =  0;} if(jn == -1){jn = ny;}
      g[k][in][jn] = gtmp[k][i][j];
    } } }

    // boundary condition
    // Half-way bounce back
    if(flag == 1){
      for(i = 0; i <= nx; i++){
//        g[2][i][   1] =-g[4][i][ 0] + cos(wav*(double)i)/3.0;
//        g[4][i][ny-1] =-g[2][i][ny] + cos(wav*(double)i)/3.0;
        g[2][i][   1] = 2.0*(    q - 1.0)                 *g[4][i][ 0]
                      - pow((2.0*q - 1.0),2)/(2.0*q + 1.0)*g[4][i][ 1]
                      + 2.0*(2.0*q - 1.0)   /(2.0*q + 1.0)*g[2][i][ 2]
                      + (3.0 - 2.0*q)/(2.0*q + 1.0)*cos(wav*(double)i)/3.0;

        g[4][i][ny-1] = 2.0*(    q - 1.0)                 *g[2][i][ny  ]
                      - pow((2.0*q - 1.0),2)/(2.0*q + 1.0)*g[2][i][ny-1]
                      + 2.0*(2.0*q - 1.0)   /(2.0*q + 1.0)*g[4][i][ny-2]
                      + (3.0 - 2.0*q)/(2.0*q + 1.0)*cos(wav*(double)i)/3.0;
      }
    }else{
      for(i = 0; i <= nx; i++){
//        g[2][i][   1] = g[4][i][ 0] + cos(wav*(double)i)/h*chi;
//        g[4][i][ny-1] = g[2][i][ny] + cos(wav*(double)i)/h*chi;

        g[2][i][   1] =                g[4][i][ 0]
         - (2.0*q - 1.0)/(2.0*q + 1.0)*g[4][i][ 1]
         + (2.0*q - 1.0)/(2.0*q + 1.0)*g[2][i][ 2]
         +           2.0/(2.0*q + 1.0)*cos(wav*(double)i)/h*chi;

        g[4][i][ny-1] =                g[2][i][ny  ]
         - (2.0*q - 1.0)/(2.0*q + 1.0)*g[2][i][ny-1]
         + (2.0*q - 1.0)/(2.0*q + 1.0)*g[4][i][ny-2]
         +           2.0/(2.0*q + 1.0)*cos(wav*(double)i)/h*chi;

      }
    } 

    // physics
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      e[i][j] = g[0][i][j];
      for( k = 1; k <= 4; k++){ e[i][j] = e[i][j] + g[k][i][j]; } 
    } }

  } //loop2
  
  err = 0.0; tmp = 0.0;
  for(i = 0; i <= nx; i++){ for(j = 1; j <= ny - 1; j++){
    err = err + pow(e[i][j] - ea[i][j],2);
    tmp = tmp + pow(ea[i][j],2);
  } }
  err = sqrt(err/tmp);

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    a[i][j]='0'; b[i][j]='0';
  } }

  emax = -999.9; emin = 999.9;
  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = e[i][j];
    if(tmp >= emax){ emax = tmp; }
    if(tmp <= emin){ emin = tmp; }
  } }

  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = e[i][j];
    if(tmp <=  emax*1.0            ){a[i][j] = '9';}
    if(tmp <= (emax*0.9 + emin*0.1)){a[i][j] = '8';}
    if(tmp <= (emax*0.8 + emin*0.2)){a[i][j] = '7';}
    if(tmp <= (emax*0.7 + emin*0.3)){a[i][j] = '6';}
    if(tmp <= (emax*0.6 + emin*0.4)){a[i][j] = '5';}
    if(tmp <= (emax*0.5 + emin*0.5)){a[i][j] = '4';}
    if(tmp <= (emax*0.4 + emin*0.6)){a[i][j] = '3';}
    if(tmp <= (emax*0.3 + emin*0.7)){a[i][j] = '2';}
    if(tmp <= (emax*0.2 + emin*0.8)){a[i][j] = '1';}
    if(tmp <= (emax*0.1 + emin*0.9)){a[i][j] = '0';}
  } }

  emax = -999.9; emin = 999.9;
  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = ea[i][j];
    if(tmp >= emax){ emax = tmp; }
    if(tmp <= emin){ emin = tmp; }
  } }

  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = ea[i][j];
    if(tmp <=  emax*1.0            ){b[i][j] = '9';}
    if(tmp <= (emax*0.9 + emin*0.1)){b[i][j] = '8';}
    if(tmp <= (emax*0.8 + emin*0.2)){b[i][j] = '7';}
    if(tmp <= (emax*0.7 + emin*0.3)){b[i][j] = '6';}
    if(tmp <= (emax*0.6 + emin*0.4)){b[i][j] = '5';}
    if(tmp <= (emax*0.5 + emin*0.5)){b[i][j] = '4';}
    if(tmp <= (emax*0.4 + emin*0.6)){b[i][j] = '3';}
    if(tmp <= (emax*0.3 + emin*0.7)){b[i][j] = '2';}
    if(tmp <= (emax*0.2 + emin*0.8)){b[i][j] = '1';}
    if(tmp <= (emax*0.1 + emin*0.9)){b[i][j] = '0';}
  } }

  if(flag == 1){ printf("Dirichlet boundary codition\n"); }
  if(flag == 2){ printf("Neumann boundary codition\n"); }
  printf("Pe = %4.1f, q = %4.2f, taug = %4.2f, wav = %5.3e, u0 = %5.3e\n", pe, q, taug, wav, u0);
  printf("Time = %d, Norm = %8.6e Error = %8.6e\n", time, norm, err);
  printf("eb[2] = %6.4e(%6.4e), et[1] = %6.4e(%6.4e)\n",e[nx/2][2],ea[nx/2][2],e[nx/2][1],ea[nx/2][1]);
  printf("eb[2] = %6.4e(%6.4e), eb[1] = %6.4e(%6.4e)\n",e[nx/2][ny-1],ea[nx/2][ny-1],e[nx/2][ny-2],ea[nx/2][ny-2]);
  printf("  (Numerical)             (Analytical)\n");

  for(j = ny - 1; j >= 1; j = j - 4){
    for(i = 1; i <= nx - 1; i = i + 4){
      printf("%c",a[i][j]);
    }
    printf("\t");
    for(i = 1; i <= nx - 1; i = i + 4){
      printf("%c",b[i][j]);
    }

    printf("\n");
  }

  } //loop1

  fp = fopen("datae","w");
  for(j = 1; j <= ny-1; j++){
    for(i = 1; i <= nx-1; i++){
      fprintf(fp," %10.8e", e[i][j]);
    }
    fprintf(fp,"\n");
  } 
  fclose(fp);
  exit(0);

  return 0;
}
