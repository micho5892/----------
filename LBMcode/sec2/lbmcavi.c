// cavity flow (compressibility error)
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
// err: relative error in the density

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

# define DIM 52 

int main(void)
{
  FILE   *fp;
  int    nx = 51, ny = 51, time = 0, loop1, loop2;
  int    i, j, k, in, jn, ip, im, jp, jm;
  double rho[DIM][DIM], u[DIM][DIM], v[DIM][DIM], un[DIM][DIM], vn[DIM][DIM], sf[DIM][DIM];
  double f[9][DIM][DIM], f0[9][DIM][DIM], ftmp[9][DIM][DIM], cx[9], cy[9];
  double ut = 0.100;
//  double ut = 0.075;
//  double ut = 0.050;
//  double ut = 0.025;
  double vt = 0.0, umax, umin, tmp, u2, nu, norm, err, tau, re = 100;
  char   a[DIM][DIM];

  // initial condition
  nu = ut*(double)(nx - 1)/re;
  tau = 3.0*nu + 0.5;

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    u[i][j] = 0.0; v[i][j] = 0.0; un[i][j] = 0.0; vn[i][j] = 0.0;
    rho[i][j] = 1.0;
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
  for(loop1 = 0; loop1 < 400; loop1++){
  for(loop2 = 0; loop2 < 500; loop2++){
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

    // propagation 
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
      ftmp[k][i][j] = f[k][i][j];
    } } }
/*
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
*/
    for(i = 1; i <= nx-1; i++){ for(j = 1; j <= ny-1; j++){ for(k = 0; k <= 8; k++){
      in = i + (int)cx[k]; jn = j + (int)cy[k];
      if(in == nx + 1){in =  0;}
      if(in ==    - 1){in = nx;}
      if(jn == ny + 1){jn =  0;}
      if(jn ==    - 1){jn = ny;}
      f[k][in][jn] = ftmp[k][i][j];
    } } }

    // boundary condition
    // Half-way bounce back
    for(j = 0; j <= ny; j++){
      jm = j - 1; if(j ==  0){jm = ny;}
      jp = j + 1; if(j == ny){jp =  0;}
      f[1][   1][j] = f[3][0][j ];
      f[5][   1][j] = f[7][0][jm];
      f[8][   1][j] = f[6][0][jp];

      f[3][nx-1][j] = f[1][nx][j ];
      f[7][nx-1][j] = f[5][nx][jp];
      f[6][nx-1][j] = f[8][nx][jm];
    }

    for(i = 0; i <= nx; i++){
      im = i - 1; if(i ==  0){im = nx;}
      ip = i + 1; if(i == nx){ip =  0;}
      f[2][i][   1] = f[4][i ][ 0];
      f[5][i][   1] = f[7][im][ 0];
      f[6][i][   1] = f[8][ip][ 0];

      rho[i][ny] = (f[0][i][ny] + f[1][i][ny] + f[3][i][ny]
             + 2.0*(f[2][i][ny] + f[5][i][ny] + f[6][i][ny]))/(1.0 + vt);
      f[4][i][ny-1] = f[2][i ][ny] - rho[i][ny]*(cx[2]*ut + cy[2]*vt)*2.0/3.0;
      f[7][i][ny-1] = f[5][ip][ny] - rho[i][ny]*(cx[5]*ut + cy[5]*vt)/6.0;
      f[8][i][ny-1] = f[6][im][ny] - rho[i][ny]*(cx[6]*ut + cy[6]*vt)/6.0;
    }
    // corner (rho = 1.0)
    u2 = ut*ut + vt*vt;      
    tmp = cx[6]*ut + cy[6]*vt;      
    f[6][nx-1][ny-1] = (1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
    tmp = cx[8]*ut + cy[8]*vt;      
    f[8][nx-1][ny-1] = (1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;

    tmp = cx[5]*ut + cy[5]*vt;      
    f[5][   1][ny-1] = (1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
    tmp = cx[7]*ut + cy[7]*vt;      
    f[7][   1][ny-1] = (1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;

    f[6][nx-1][   1] = 1.0/36.0; f[8][nx-1][   1] = 1.0/36.0;
    f[5][   1][   1] = 1.0/36.0; f[7][   1][   1] = 1.0/36.0;

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
  
  tmp = 0.0;
  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = tmp + rho[i][j];
  } }
  tmp = tmp/(nx - 1)/(ny - 1);

  err = 0.0;
  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    err = err + pow((rho[i][j] - tmp),2);
  } }
  err = sqrt(err)/(nx - 1)/(ny - 1);
  err = err/tmp;

  printf("Time = %d, Norm = %8.6e, Error = %8.6e\n", time, norm, err);

  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    a[i][j]='0';
  } }

  umax = -999.9; umin = 999.9;
  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = sqrt(pow(u[i][j],2) + pow(v[i][j],2));
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

  printf("Re = %4.2f, ut = %5.3f, tau = %5.3f\n", re, ut, tau);

  for(j = ny - 1; j >= 1; j = j - 2){ for(i = 1; i <= nx - 1; i = i + 2){
    printf("%c",a[i][j]);
    }
    printf("\n");
  }

  if(norm < 0.000000000001 && time > 10000){ exit(0); }

  } //loop1

  fp = fopen("error","w");
  fprintf(fp,"%15.8e, %15.8e\n", ut, err);
  fclose(fp);

// stream function
  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
   sf[i][j] = 0.0;
  } }

  for(i = 0; i <= nx; i++){ for(j = 2; j <= ny; j++){
    sf[i][j] = (u[i][j] + 4.*u[i][j-1] + u[i][j-2])/3.0 + sf[i][j-2];
  } }
  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    sf[i][j] = sf[i][j]/ut/(double)(nx-1);
  } }

  umax = -999.9; umin = 999.9;
  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = sf[i][j];
    if(tmp >= umax){umax = tmp;}
    if(tmp <= umin){umin = tmp;}
  } }
  printf("stream function: %10.8e, %10.8e", umax, umin);

  fp = fopen("datau","w");
    for(j = 1; j <= ny-1; j++){ for(i = 1; i <= nx-1; i++){
      fprintf(fp," %10.8e", u[i][j])/ut;
    }
  fprintf(fp,"\n");
  } 
  fclose(fp);

  fp = fopen("datav","w");
    for(j = 1; j <= ny-1; j++){ for(i = 1; i <= nx-1; i++){
      fprintf(fp," %10.8e", v[i][j]/ut);
    }
  fprintf(fp,"\n");
  } 
  fclose(fp);

  fp = fopen("datas","w");
    for(j = 1; j <= ny-1; j++){ for(i = 1; i <= nx-1; i++){
      fprintf(fp," %10.8e", sf[i][j]);
    }
  fprintf(fp,"\n");
  } 
  fclose(fp);

  return 0;
}
