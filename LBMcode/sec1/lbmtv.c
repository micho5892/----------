// Taylor vortex flow 
// Lattice Boltzmann Method. The source code is written in C programming language.
// Copyright @2021 Takeshi Seta All Rights Reserved.
//
// 6  2  5
//    |
// 3--0--1
//    |
// 7  4  8
//
// rho : density (numerical solution)
// u   : horizontal velocity component (numerical solution)
// v   : vertical velocity component (numerical solution)
// rhoe: density (analytical solution)
// ue  : horizontal velocity component (analytical solution)
// ve  : vertical velocity component (analytical solution)
// nx  : number of grid points (x-axis)
// ny  : number of grid points (y-axis)
// u0  : characteristic velocity
// f   : distribution function
// f0  : equilibrium distribution function
// cx  : discrete velocity (x-axis)
// cy  : discrete velocity (y-axis)
// tau : relaxation time
// nu  : kinematic viscosity
// erru: relative error in u (horizontal velocity)
// errv: relative error in v (vertical velocity)
// errp: relative error in p (pressure)
// M_PI: Pi

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

# define DIM 80

int main(void)
{
  FILE   *fp;
  int    nx, ny, time = 0, loop1, loop2;
  int    i, j, k, in, jn;
  double  rho[DIM][DIM],  u[DIM][DIM],  v[DIM][DIM];
  double rhoe[DIM][DIM], ue[DIM][DIM], ve[DIM][DIM];
  double f[9][DIM][DIM], f0[9][DIM][DIM], ftmp[9][DIM][DIM], cx[9], cy[9];
  double u0 = 0.01, tmp, u2, nu, erru, errv, errp, tau;

  // initial condition
  tau = 0.8;
  nu = (tau - 0.5)/3.0;
//  nx = 80; ny = nx;
//  nx = 40; ny = nx;
//  nx = 20; ny = nx;
//  nx = 10; ny = nx;
  nx = 5; ny = nx;

  for(i = 0; i < nx; i++){ for(j = 0; j < ny; j++){
    u[i][j] = -u0*cos(2.0*M_PI*(double)i/(double)nx)*sin(2.0*M_PI*(double)j/(double)ny);
    v[i][j] =  u0*sin(2.0*M_PI*(double)i/(double)nx)*cos(2.0*M_PI*(double)j/(double)ny);
    rho[i][j] = -0.75*u0*u0*(sin(4.0*M_PI*(double)i/(double)nx)
                           + cos(4.0*M_PI*(double)j/(double)ny)) + 1.0;
  } }

  cx[0] = 0.0; cy[0] = 0.0;
  cx[1] = 1.0; cy[1] = 0.0;  cx[2] = 0.0; cy[2] = 1.0;
  cx[3] =-1.0; cy[3] = 0.0;  cx[4] = 0.0; cy[4] =-1.0;
  cx[5] = 1.0; cy[5] = 1.0;  cx[6] =-1.0; cy[6] = 1.0;
  cx[7] =-1.0; cy[7] =-1.0;  cx[8] = 1.0; cy[8] =-1.0;

  for(i = 0; i < nx; i++){ for(j = 0; j < ny; j++){
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

  for(i = 0; i < nx; i++){ for(j = 0; j < ny; j++){ for(k = 0; k <= 8; k++){
    f[k][i][j] = f0[k][i][j];
  } } }

// calculation start
  for(loop1 = 0; loop1 < nx; loop1++){
  for(loop2 = 0; loop2 < nx; loop2++){
    time++;

    for(i = 0; i < nx; i++){ for(j = 0; j < ny; j++){
      ue[i][j] = -u0*cos(2.0*M_PI*(double)i/(double)nx)*sin(2.0*M_PI*(double)j/(double)ny)
               * exp(-nu*time*(pow(2.0*M_PI/(double)nx,2)+pow(2.0*M_PI/(double)nx,2)));
      ve[i][j] =  u0*sin(2.0*M_PI*(double)i/(double)nx)*cos(2.0*M_PI*(double)j/(double)ny)
               * exp(-nu*time*(pow(2.0*M_PI/(double)nx,2)+pow(2.0*M_PI/(double)nx,2)));
    rhoe[i][j] = 1.0 - 0.75*u0*u0*(sin(4.0*M_PI*(double)i/(double)nx)
                                 + cos(4.0*M_PI*(double)j/(double)ny))
               * exp(-2.0*nu*time*(pow(2.0*M_PI/(double)nx,2)+pow(2.0*M_PI/(double)nx,2)));
    } }

    // collision
    for(i = 0; i < nx; i++){ for(j = 0; j < ny; j++){
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

    for(i = 0; i < nx; i++){ for(j = 0; j < ny; j++){ for(k = 0; k <= 8; k++){
      f[k][i][j] = f[k][i][j] - (f[k][i][j] - f0[k][i][j])/tau;
    } } }

    // propagation 
    for(i = 0; i < nx; i++){ for(j = 0; j < ny; j++){ for(k = 0; k <= 8; k++){
      ftmp[k][i][j] = f[k][i][j];
    } } }

    for(i = 0; i < nx; i++){ for(j = 0; j < ny; j++){ for(k = 0; k <= 8; k++){
      in = i + (int)cx[k]; jn = j + (int)cy[k];
      if(in == nx){in =      0;}
      if(in == -1){in = nx - 1;}
      if(jn == ny){jn =      0;}
      if(jn == -1){jn = ny - 1;}
      f[k][in][jn] = ftmp[k][i][j];
    } } }

    // physics
    for(i = 0; i < nx; i++){ for(j = 0; j < ny; j++){
      rho[i][j] = f[0][i][j]; u[i][j] = 0; v[i][j] = 0;
      for( k = 1; k <= 8; k++){
        rho[i][j] = rho[i][j] + f[k][i][j];
          u[i][j] =   u[i][j] + f[k][i][j]*cx[k];
          v[i][j] =   v[i][j] + f[k][i][j]*cy[k];
      } 
      u[i][j] = u[i][j]/rho[i][j];
      v[i][j] = v[i][j]/rho[i][j];
    } }

  } //loop2
  } //loop1

  erru = 0.0; tmp = 0.0;
  for(i = 0; i < nx; i++){ for(j = 0; j < ny; j++){
   erru = erru + pow((ue[i][j] - u[i][j]),2); tmp = tmp + pow(ue[i][j],2);
  } }
  erru = sqrt(erru/tmp);

  errv = 0.0; tmp = 0.0;
  for(i = 0; i < nx; i++){ for(j = 0; j < ny; j++){
   errv = errv + pow((ve[i][j] - v[i][j]),2); tmp = tmp + pow(ve[i][j],2);
  } }
  errv = sqrt(errv/tmp);

  errp = 0.0; tmp = 0.0;
  for(i = 0; i < nx; i++){ for(j = 0; j < ny; j++){
   errp = errp + pow((rhoe[i][j] - rho[i][j]),2); tmp = tmp + pow(rhoe[i][j],2);
  } }
  errp = sqrt(errp/tmp);

  printf("Time = %f, nx = %d, tau= %f\n",time*u0/(double)nx, nx, tau);
  printf("Erru = %10.8e, Errv = %10.8e, Errp = %10.8e\n",erru, errv, errp);

  fp = fopen("error","w");
    fprintf(fp,"u %15.8e\n", erru);
    fprintf(fp,"v %15.8e\n", errv);
    fprintf(fp,"p %15.8e\n", errp);
  fclose(fp);

  fp = fopen("datautv","w");
    for(i = 0; i < nx; i++){ for(j = 0; j < ny; j++){
      fprintf(fp," %10.8e", u[i][j]);
    }
  fprintf(fp,"\n");
  } 
  fclose(fp);

  fp = fopen("datavtv","w");
    for(i = 0; i < nx; i++){ for(j = 0; j < ny; j++){
      fprintf(fp," %10.8e", v[i][j]);
    }
  fprintf(fp,"\n");
  } 
  fclose(fp);

  fp = fopen("datartv","w");
    for(i = 0; i < nx; i++){ for(j = 0; j < ny; j++){
      fprintf(fp," %e", rho[i][j]/3.0);
    }
  fprintf(fp,"\n");
  } 
  fclose(fp);

  fp = fopen("datautve","w");
    for(i = 0; i < nx; i++){ for(j = 0; j < ny; j++){
      fprintf(fp," %10.8e", ue[i][j]);
    }
  fprintf(fp,"\n");
  } 
  fclose(fp);

  fp = fopen("datavtve","w");
    for(i = 0; i < nx; i++){ for(j = 0; j < ny; j++){
      fprintf(fp," %10.8e", ve[i][j]);
    }
  fprintf(fp,"\n");
  } 
  fclose(fp);

  fp = fopen("datartve","w");
    for(i = 0; i < nx; i++){ for(j = 0; j < ny; j++){
      fprintf(fp," %e", rhoe[i][j]/3.0);
    }
  fprintf(fp,"\n");
  } 
  fclose(fp);

  return 0;
}
