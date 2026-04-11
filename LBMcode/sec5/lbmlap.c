// Laplace's law  
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
// rho : mean density
// u   : horizontal velocity component (time = n)
// v   : vertical velocity component (time = n)
// un  : horizontal velocity component (time = n -1)
// vn  : vertical velocity component (time = n - 1)
// phi : order parameter
// che : chemical potential
// fx  : body force (x-axis)
// fy  : body force (y-axis)
//
// gamma: coefficient for the mobility
// sig : surface tension
// wid : interface width
// phi0: minimum value of double-well potential
// beta: coefficient for the chemical potentioal
// kap : coefficient for the surface tension
// nx  : number of grid points (x-axis)
// ny  : number of grid points (y-axis)
//
// f   : distribution function (mean density, velocity)
// f0  : equilibrium distribution function (mean density, velocity)
// g   : distribution function (order parameter)
// g0  : equilibrium distribution function
// fi  : forcing term
//
// cx  : discrete velocity (x-axis)
// cy  : discrete velocity (y-axis)
// re  : Reynolds number
// err : relative error in the density
// tauf: relaxation time (f)
// taug: relaxation time (g)

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

# define DIM 51

int main(void)
{
  FILE   *fp;
  int    nx = 50, ny = 50, time = 0, loop1, loop2;
  int    i, j, k, in, jn, ip, im, jp, jm;
  double rho[DIM][DIM],  u[DIM][DIM], v[DIM][DIM], phi[DIM][DIM], che[DIM][DIM];
  double  un[DIM][DIM], vn[DIM][DIM], fx[DIM][DIM],  fy[DIM][DIM];
  double f[9][DIM][DIM], f0[9][DIM][DIM], ftmp[9][DIM][DIM], fi[9][DIM][DIM], cx[9], cy[9];
  double g[9][DIM][DIM], g0[9][DIM][DIM], gtmp[9][DIM][DIM];
  double umax, umin, tmp, u2, norm, tauf, taug;
  double gamma, sig, wid, phi0, beta, kap, chex, chey, rhox, rhoy, tmp1, tmp2;
  char   a[DIM][DIM], b[DIM][DIM];

  // initial condition

  tauf = 0.7;
  taug = 0.7;

  gamma = 10.0; wid = 5.0; phi0 = 1.0;
//  sig = 0.0010;
  sig = 0.0001;
  beta  = 3.0/4.0*sig/wid*pow(phi0,4);
  kap   = 3.0/8.0*sig*wid/pow(phi0,2);

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    u[i][j] = 0.0; v[i][j] = 0.0; un[i][j] = 0.0; vn[i][j] = 0.0; rho[i][j] = 1.0;
    tmp = sqrt(pow(((double)i - (double)nx*0.5),2)
             + pow(((double)j - (double)ny*0.5),2));
    phi[i][j] = -phi0*tanh(2.0*(tmp - (double)nx*0.25)/wid);
  } }

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    ip = i + 1; if(i == nx){ ip =  0; }
    im = i - 1; if(i ==  0){ im = nx; }
    jp = j + 1; if(j == ny){ jp =  0; }
    jm = j - 1; if(j ==  0){ jm = ny; }

    tmp = phi[ip][j] + phi[im][j] + phi[i][jp] + phi[i][jm] - 4.0*phi[i][j];
    che[i][j] = 4.*beta*(pow(phi[i][j],2) - pow(phi0,2))*phi[i][j] - kap*tmp;
  } }

  cx[0] = 0.0; cy[0] = 0.0;
  cx[1] = 1.0; cy[1] = 0.0;  cx[2] = 0.0; cy[2] = 1.0;
  cx[3] =-1.0; cy[3] = 0.0;  cx[4] = 0.0; cy[4] =-1.0;
  cx[5] = 1.0; cy[5] = 1.0;  cx[6] =-1.0; cy[6] = 1.0;
  cx[7] =-1.0; cy[7] =-1.0;  cx[8] = 1.0; cy[8] =-1.0;

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    u2 = u[i][j]*u[i][j] + v[i][j]*v[i][j];      
    f0[0][i][j] = rho[i][j]*(1.0 -3.0/2.0*u2 - 15.0/4.0*phi[i][j]*che[i][j])*4.0/9.0;
    g0[0][i][j] = phi[i][j] - 5.0/9.0*gamma*che[i][j];

    for(k = 1; k <= 4; k++){
      tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
      f0[k][i][j] = rho[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2 + 3.0*phi[i][j]*che[i][j])/9.0;
      g0[k][i][j] = (gamma*che[i][j] + 3.0*phi[i][j]*tmp)/9.0;
    }

    for(k = 5; k <= 8; k++){
      tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
      f0[k][i][j] = rho[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2 + 3.0*phi[i][j]*che[i][j])/36.0;
      g0[k][i][j] = (gamma*che[i][j] + 3.0*phi[i][j]*tmp)/36.0;
    }

  } }

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
    f[k][i][j] = f0[k][i][j]; g[k][i][j] = g0[k][i][j];
  } } }

// calculation start
  for(loop1 = 0; loop1 < 1000; loop1++){
  for(loop2 = 0; loop2 < 500; loop2++){
    time++;

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      un[i][j] = u[i][j]; vn[i][j] = v[i][j];
    } }

  // collision
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      u2 = u[i][j]*u[i][j] + v[i][j]*v[i][j];      
      f0[0][i][j] = rho[i][j]*(1.0 -3.0/2.0*u2 -15.0/4.0*phi[i][j]*che[i][j])*4.0/9.0;
    g0[0][i][j] = phi[i][j] - 5.0/9.0*gamma*che[i][j];

      for(k = 1; k <= 4; k++){
        tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
        f0[k][i][j] = rho[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2 +3.0*phi[i][j]*che[i][j])/9.0;
      g0[k][i][j] = (gamma*che[i][j] + 3.0*phi[i][j]*tmp)/9.0;
      }  

      for(k = 5; k <= 8; k++){
        tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
        f0[k][i][j] = rho[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2 +3.0*phi[i][j]*che[i][j])/36.0;
      g0[k][i][j] = (gamma*che[i][j] + 3.0*phi[i][j]*tmp)/36.0;
      }
    } }

// SRT collision
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      for(k = 0; k <= 8; k++){
        f[k][i][j] = f[k][i][j] - (f[k][i][j] - f0[k][i][j])/tauf;
        g[k][i][j] = g[k][i][j] - (g[k][i][j] - g0[k][i][j])/taug;
      }
    } }

//  forcing
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      fx[i][j] = 0.0; fy[i][j] = 0.0;
      for(k = 0; k <= 8; k++){
        fi[k][i][j] = 0.0;
      }
    } }

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      ip = i + 1; if(i == nx){ ip =  0; }
      im = i - 1; if(i ==  0){ im = nx; }
      jp = j + 1; if(j == ny){ jp =  0; }
      jm = j - 1; if(j ==  0){ jm = ny; }


      fx[i][j] = che[i][j] * (phi[ip][j ] - phi[im][j ])*0.5;
      fy[i][j] = che[i][j] * (phi[i ][jp] - phi[i ][jm])*0.5;

      u2 = u[i][j]*fx[i][j] + v[i][j]*fy[i][j];

      fi[0][i][j] = -4.0/3.0*(1.0 - 0.5/tauf)*u2;
//      fi[0][i][j] = 0.0;

     for(k = 1; k <= 4; k++){
       tmp = cx[k]*fx[i][j] + cy[k]*fy[i][j];

       tmp1 = u[i][j]*fx[i][j]*cx[k]*cx[k]
            + u[i][j]*fy[i][j]*cx[k]*cy[k]
            + v[i][j]*fx[i][j]*cy[k]*cx[k]
            + v[i][j]*fy[i][j]*cy[k]*cy[k];
       fi[k][i][j] = (1.0 - 0.5/tauf)*(3.0*tmp + 9.0*tmp1 - 3.0*u2)/9.0;
      }
     for(k = 5; k <= 8; k++){
       tmp = cx[k]*fx[i][j] + cy[k]*fy[i][j];
       tmp1 = u[i][j]*fx[i][j]*cx[k]*cx[k]
            + u[i][j]*fy[i][j]*cx[k]*cy[k]
            + v[i][j]*fx[i][j]*cy[k]*cx[k]
            + v[i][j]*fy[i][j]*cy[k]*cy[k];
       fi[k][i][j] = (1.0 - 0.5/tauf)*(3.0*tmp + 9.0*tmp1 - 3.0*u2)/36.0;
      }
    } }

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
      f[k][i][j] = f[k][i][j] + fi[k][i][j];
    } } }

    // propagation 
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
      ftmp[k][i][j] = f[k][i][j];
      gtmp[k][i][j] = g[k][i][j];
    } } }

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
      in = i + (int)cx[k]; jn = j + (int)cy[k];
      if(in == nx + 1){in =  0;}
      if(in ==    - 1){in = nx;}
      if(jn == ny + 1){jn =  0;}
      if(jn ==    - 1){jn = ny;}
      f[k][in][jn] = ftmp[k][i][j];
      g[k][in][jn] = gtmp[k][i][j];
    } } }

    // physics
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      rho[i][j] = f[0][i][j]; u[i][j] = 0; v[i][j] = 0;
      phi[i][j] = g[0][i][j];
      for( k = 1; k <= 8; k++){
        rho[i][j] = rho[i][j] + f[k][i][j];
          u[i][j] =   u[i][j] + f[k][i][j]*cx[k];
          v[i][j] =   v[i][j] + f[k][i][j]*cy[k];
        phi[i][j] = phi[i][j] + g[k][i][j];
      } 
      u[i][j] = u[i][j]/rho[i][j];
      v[i][j] = v[i][j]/rho[i][j];
    } }

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      u[i][j] = u[i][j] + fx[i][j]*0.5;
      v[i][j] = v[i][j] + fy[i][j]*0.5;
    } }

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    ip = i + 1; if(i == nx){ ip =  0; }
    im = i - 1; if(i ==  0){ im = nx; }
    jp = j + 1; if(j == ny){ jp =  0; }
    jm = j - 1; if(j ==  0){ jm = ny; }

    tmp = phi[ip][j] + phi[im][j] + phi[i][jp] + phi[i][jm] - 4.0*phi[i][j];
    che[i][j] = 4.*beta*(pow(phi[i][j],2) - pow(phi0,2))*phi[i][j] - kap*tmp;
  } }

    norm = 0.0;
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      tmp = sqrt(pow(u[i][j] - un[i][j],2) + pow(v[i][j] - vn[i][j],2));
      if(tmp > norm){ norm = tmp; }
    } }

  } //loop2
  
  printf("Time = %d, Norm = %8.6e \n", time, norm);
  printf("laplace's law  : %10.8f, %10.8f\n", sig/(double)nx/0.25, (rho[nx/2][ny/2]-rho[0][0])/3.0 );

  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    a[i][j]='0'; b[i][j]='0';
  } }

  umax = -999.9; umin = 999.9;
  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = phi[i][j];
    if(tmp >= umax){umax = tmp;}
    if(tmp <= umin){umin = tmp;}
  } }

  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = phi[i][j];
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
  printf("index function : %10.8f, %10.8f\n", umax, umin);


  umax = -999.9; umin = 999.9;
  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = sqrt(u[i][j]*u[i][j] + v[i][j]*v[i][j]);
    if(tmp >= umax){umax = tmp;}
    if(tmp <= umin){umin = tmp;}
  } }
  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = sqrt(u[i][j]*u[i][j] + v[i][j]*v[i][j]);
    if(tmp <=  umax*1.0            ){b[i][j] = '9';}
    if(tmp <= (umax*0.9 + umin*0.1)){b[i][j] = '8';}
    if(tmp <= (umax*0.8 + umin*0.2)){b[i][j] = '7';}
    if(tmp <= (umax*0.7 + umin*0.3)){b[i][j] = '6';}
    if(tmp <= (umax*0.6 + umin*0.4)){b[i][j] = '5';}
    if(tmp <= (umax*0.5 + umin*0.5)){b[i][j] = '4';}
    if(tmp <= (umax*0.4 + umin*0.6)){b[i][j] = '3';}
    if(tmp <= (umax*0.3 + umin*0.7)){b[i][j] = '2';}
    if(tmp <= (umax*0.2 + umin*0.8)){b[i][j] = '1';}
    if(tmp <= (umax*0.1 + umin*0.9)){b[i][j] = '0';}
  } }
  printf("velocity       : %10.8f, %10.8f\n", umax, umin);

  umax = -999.9; umin = 999.9;
  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = rho[i][j];
    if(tmp >= umax){umax = tmp;} if(tmp <= umin){umin = tmp;}
  } }
  printf("density        : %10.8f, %10.8f\n", umax, umin);

  for(j = ny - 1; j >= 1; j = j - 2){
     for(i = 1; i <= nx - 1; i = i + 2){
    printf("%c",a[i][j]);
   }
   printf("\t");
   for(i = 1; i <= nx - 1; i = i + 2){
    printf("%c",b[i][j]);
   }
    printf("\n");
  }

  if(norm < 0.0000000001 && time > 10000){
    fp = fopen("datalap","w");
    fprintf(fp,"%10.8e,%10.8e", (double)nx*0.5, (rho[nx/2][ny/2]-rho[0][0])/3.0 );
    fclose(fp);

    fp = fopen("dataphi2D","w");
      for(i = 0; i <= nx; i++){
        fprintf(fp," %10.8e", phi[i][ny/2]);
      } 
    fclose(fp);

    fp = fopen("datau","w");
      for(j = 1; j <= ny-1; j++){ for(i = 1; i <= nx-1; i++){
        fprintf(fp," %10.8e", u[i][j]);
      }
    fprintf(fp,"\n");
    } 
    fclose(fp);

    fp = fopen("datav","w");
      for(j = 0; j <= ny; j++){ for(i = 0; i <= nx; i++){
        fprintf(fp," %10.8e", v[i][j]);
      }
    fprintf(fp,"\n");
    } 
    fclose(fp);

    fp = fopen("datarho","w");
      for(j = 0; j <= ny; j++){ for(i = 0; i <= nx; i++){
        fprintf(fp," %10.8e", rho[i][j]);
      }
    fprintf(fp,"\n");
    } 
    fclose(fp);

    fp = fopen("dataphi","w");
      for(j = 0; j <= ny; j++){ for(i = 0; i <= nx; i++){
        fprintf(fp," %10.8e", phi[i][j]);
      }
    fprintf(fp,"\n");
    } 
    fclose(fp);

    exit(0);
  }

  } //loop1

//  printf("stream function: %10.8f, %10.8f", umax, umin);
  return 0;
}
