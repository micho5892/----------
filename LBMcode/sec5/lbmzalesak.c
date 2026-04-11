// Zalesak's Disk
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
// rho : density
// u   : horizontal velocity component (time = n)
// v   : vertical velocity component (time = n)
// un  : horizontal velocity component (time = n -1)
// vn  : vertical velocity component (time = n - 1)
// phi : order parameter (time = n)
// phin: order parameter (time = n - 1)
// che : chemical potential
//
// gamma: coefficient for the mobility
// sig : surface tension
// wid : interface width
// phi0: minimum value of double-well potential
// beta: coefficient for the chemical potentioal
// kap : coefficient for the surface tension
// nx  : number of grid points (x-axis)
// nx  : number of grid points (x-axis)
// ny  : number of grid points (y-axis)
//
// g   : distribution function
// g0  : equilibrium distribution function
// cx  : discrete velocity (x-axis)
// cy  : discrete velocity (y-axis)
// tau : relaxation time
//
// pe  : Peclet number
// mo  : mobility
// mc  : transformation matrix
// mi  : inverse matrix of mc
// sc  : relaxation matrix for distribution function
// sf  : relaxation matrix for force
// flag = 1 (SRT collision) 
// flag = 2 (SRT collision + force)
// flag = 3 (MRT collision) 
// flag = 4 (MRT collision + force)

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

# define DIM 51

int main(void)
{
  FILE   *fp;
  int    nx = 50, ny = 50, loop1, loop2;
  int    i, j, k, m, in, jn, ip, im, jp, jm, flag;
  double g[9][DIM][DIM], g0[9][DIM][DIM], gtmp[9][DIM][DIM], cx[9], cy[9];
  double t[9][DIM][DIM], t0[9][DIM][DIM], gm[9][DIM][DIM];
  double mc[9][9], mi[9][9], sc[9][9], sf[9][9];
  double u[DIM][DIM], v[DIM][DIM], che[DIM][DIM], phi[DIM][DIM];
  double un[DIM][DIM], vn[DIM][DIM], phin[DIM][DIM];
  double time = 0.0, umax, umin, tmp, u2, tau, pe, mo;
  double gamma, sig, wid, phi0, beta, kap, chex, chey, u0;
  double s0,s1, s2, s3, s4, s5, s6, s7, s8;

  char   a[DIM][DIM];

  flag = 1; // SRT 
//  flag = 2; // SRT + force
//  flag = 3; // MRT 
//  flag = 4; // MRT + force

  // initial condition

  tau = 0.75;
  u0 = 0.04;
  sig = 0.04; wid = 2.0; phi0 = 1.0;
  beta  = 3.0/4.0*sig/wid*pow(phi0,4);
  kap   = 3.0/8.0*sig*wid/pow(phi0,2);
  pe = 400.0;
  mo = u0*wid/pe/beta/4.0;
  gamma = mo/(tau - 0.5)*3.0;

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    phi[i][j] = -1.0; u[i][j] = 0.0; v[i][j] = 0.0;
    un[i][j] = 0.0; vn[i][j] = 0.0;
  } }

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    tmp = sqrt( pow(((double)i - (double)nx*0.5),2)
              + pow(((double)j - (double)ny*0.5),2) );
    if(tmp <= (double)nx*0.4){ phi[i][j] = 1.0; }
  } }

  for(i = 93*nx/200; i <= 107*nx/200; i++){ for(j = 2; j <= ny/2; j++){
    phi[i][j] = -1.0;
  } }

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    phin[i][j] = phi[i][j];
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


  s0 = 1.0  ; s1 = 1.3; s2 = 1.3; s3 = 1/tau; s4 = 1.3; s5 = 1./tau;
  s6 = 1.3  ; s7 = 1.0; s8 = 1.0;

  for(i = 0; i <= 8; i++){ for(j = 0; j <= 8; j++){ sc[i][j] = 0.0; } }

  sc[0][0] = s0; sc[1][1] = s1; sc[2][2] = s2;
  sc[3][3] = s3; sc[4][4] = s4; sc[5][5] = s5;
  sc[6][6] = s6; sc[7][7] = s7; sc[8][8] = s8;

  for(i = 0; i <= 8; i++){ for(j = 0; j <= 8; j++){ sf[i][j] = 0.0; } }

  sf[0][0] = 1.0 - s0/2.0; sf[1][1] = 1.0 - s1/2.0; sf[2][2] = 1.0 - s2/2.0;
  sf[3][3] = 1.0 - s3/2.0; sf[4][4] = 1.0 - s4/2.0; sf[5][5] = 1.0 - s5/2.0;
  sf[6][6] = 1.0 - s6/2.0; sf[7][7] = 1.0 - s7/2.0; sf[8][8] = 1.0 - s8/2.0;

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    u2 = u[i][j]*u[i][j] + v[i][j]*v[i][j];      
    g0[0][i][j] = phi[i][j] - 5.0/9.0*gamma*che[i][j];

    for(k = 1; k <= 4; k++){
      tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
      g0[k][i][j] = (gamma*che[i][j] + 3.0*phi[i][j]*tmp)/9.0;
    }

    for(k = 5; k <= 8; k++){
      tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
      g0[k][i][j] = (gamma*che[i][j] + 3.0*phi[i][j]*tmp)/36.0;
    }
  } }

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
    g[k][i][j] = g0[k][i][j];
  } } }

// calculation start
  for(loop1 = 0; loop1 < 50; loop1++){
  for(loop2 = 0; loop2 < 50; loop2++){
    time++;

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      u[i][j] =-u0*M_PI*((double)j/(double)ny - 0.5);
      v[i][j] = u0*M_PI*((double)i/(double)ny - 0.5);
    } }

  // collision
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      u2 = u[i][j]*u[i][j] + v[i][j]*v[i][j];      
      g0[0][i][j] = phi[i][j] - 5.0/3.0*gamma*che[i][j];

      for(k = 1; k <= 4; k++){
        tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
        g0[k][i][j] = (gamma*che[i][j] + phi[i][j]*tmp)/3.0;
      }  

      for(k = 5; k <= 8; k++){
        tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
        g0[k][i][j] = (gamma*che[i][j] + phi[i][j]*tmp)/12.0;
      }
    } }

    if(flag == 1){
// SRT collision
      for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
        g[k][i][j] = g[k][i][j] - (g[k][i][j] - g0[k][i][j])/tau;
      } } }
    } else if(flag == 2){
// SRT collision + force
      for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
        g[k][i][j] = g[k][i][j] - (g[k][i][j] - g0[k][i][j])/tau;
      } } }

      for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
        for(k = 1; k <= 4; k++){
          g[k][i][j] = g[k][i][j] + (1.0 - 0.5/tau)*
                       (cx[k]*(phi[i][j]*u[i][j] - phin[i][j]*un[i][j])
                      + cy[k]*(phi[i][j]*v[i][j] - phin[i][j]*vn[i][j]))/3.0;
        }
        for(k = 5; k <= 8; k++){
          g[k][i][j] = g[k][i][j] + (1.0 - 0.5/tau)*
                       (cx[k]*(phi[i][j]*u[i][j] - phin[i][j]*un[i][j])
                      + cy[k]*(phi[i][j]*v[i][j] - phin[i][j]*vn[i][j]))/12.0;
        }
      } }
   } else if(flag == 3){

// MRT collision
     for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
        t[k][i][j] = 0.0; t0[k][i][j] = 0.0; gtmp[k][i][j] = 0.0; gm[k][i][j] = 0.0;
      } } }

      for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
// transformation
        for(k = 0; k <= 8; k++){ for(m = 0; m <= 8; m++){
          t[k][i][j] = t[k][i][j] + mc[k][m] * g[m][i][j];
         t0[k][i][j] =t0[k][i][j] + mc[k][m] *g0[m][i][j];
        } }
// collision
        for(k = 0; k <= 8; k++){ for(m = 0; m <= 8; m++){
          gtmp[k][i][j] = gtmp[k][i][j] + sc[k][m] * (t[m][i][j] - t0[m][i][j]);
        } }
        for(k = 0; k <= 8; k++){
          t[k][i][j] = t[k][i][j] - gtmp[k][i][j];
        }
// inverse
        for(k = 0; k <= 8; k++){ for(m = 0; m <= 8; m++){
          gm[k][i][j] = gm[k][i][j] + mi[k][m] * t[m][i][j];
        } }

        for(k = 0; k <= 8; k++){
          g[k][i][j] = gm[k][i][j];
        }
      } }

   } else if(flag == 4){
// MRT collision + force
     for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
        t[k][i][j] = 0.0; t0[k][i][j] = 0.0; gtmp[k][i][j] = 0.0; gm[k][i][j] = 0.0;
      } } }

      for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
// transformation
        for(k = 0; k <= 8; k++){ for(m = 0; m <= 8; m++){
          t[k][i][j] = t[k][i][j] + mc[k][m] * g[m][i][j];
         t0[k][i][j] =t0[k][i][j] + mc[k][m] *g0[m][i][j];
        } }
// collision
        for(k = 0; k <= 8; k++){ for(m = 0; m <= 8; m++){
          gtmp[k][i][j] = gtmp[k][i][j] + sc[k][m] * (t[m][i][j] - t0[m][i][j]);
        } }
        for(k = 0; k <= 8; k++){
          t[k][i][j] = t[k][i][j] - gtmp[k][i][j];
        }
// inverse
        for(k = 0; k <= 8; k++){ for(m = 0; m <= 8; m++){
          gm[k][i][j] = gm[k][i][j] + mi[k][m] * t[m][i][j];
        } }

        for(k = 0; k <= 8; k++){
          g[k][i][j] = gm[k][i][j];
        }
      } }

// force
     for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
        t[k][i][j] = 0.0; t0[k][i][j] = 0.0; gtmp[k][i][j] = 0.0; gm[k][i][j] = 0.0;
      } } }

      for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
        for(k = 1; k <= 4; k++){
           t0[k][i][j] = (cx[k]*(phi[i][j]*u[i][j] - phin[i][j]*un[i][j])
                        + cy[k]*(phi[i][j]*v[i][j] - phin[i][j]*vn[i][j]))/3.0;
        }
        for(k = 5; k <= 8; k++){
          t0[k][i][j] = (cx[k]*(phi[i][j]*u[i][j] - phin[i][j]*un[i][j])
                       + cy[k]*(phi[i][j]*v[i][j] - phin[i][j]*vn[i][j]))/12.0;
        }
      } }

      for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
// transformation
        for(k = 0; k <= 8; k++){ for(m = 0; m <= 8; m++){
          t[k][i][j] = t[k][i][j] + mc[k][m] * t0[m][i][j];
        } }
// collision
        for(k = 0; k <= 8; k++){ for(m = 0; m <= 8; m++){
          gtmp[k][i][j] = gtmp[k][i][j] + sf[k][m] * t[m][i][j];
        } }
        for(k = 0; k <= 8; k++){
          t[k][i][j] = t[k][i][j] + gtmp[k][i][j];
        }
// inverse
        for(k = 0; k <= 8; k++){ for(m = 0; m <= 8; m++){
          gm[k][i][j] = gm[k][i][j] + mi[k][m] * t[m][i][j];
        } }

        for(k = 0; k <= 8; k++){
          g[k][i][j] = g[k][i][j] + gm[k][i][j];
        }

      } }

    }

    // propagation 
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
      gtmp[k][i][j] = g[k][i][j];
    } } }

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
      in = i + (int)cx[k]; jn = j + (int)cy[k];
      if(in == nx + 1){in =  0;}
      if(in ==    - 1){in = nx;}
      if(jn == ny + 1){jn =  0;}
      if(jn ==    - 1){jn = ny;}
      g[k][in][jn] = gtmp[k][i][j];
    } } }

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      phin[i][j] = phi[i][j]; un[i][j] = u[i][j]; vn[i][j] = v[i][j];
    } }

    // physics
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      phi[i][j] = g[0][i][j];
      for( k = 1; k <= 8; k++){
        phi[i][j] = phi[i][j] + g[k][i][j];
      } 
    } }

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    ip = i + 1; if(i == nx){ ip =  0; }
    im = i - 1; if(i ==  0){ im = nx; }
    jp = j + 1; if(j == ny){ jp =  0; }
    jm = j - 1; if(j ==  0){ jm = ny; }

    tmp = phi[ip][j] + phi[im][j] + phi[i][jp] + phi[i][jm] - 4.0*phi[i][j];
    che[i][j] = 4.*beta*(pow(phi[i][j],2) - pow(phi0,2))*phi[i][j] - kap*tmp;
  } }

  } //loop2
  
  printf("Time = %f\n", time);


  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    a[i][j]='0'; 
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

  for(j = ny - 1; j >= 1; j=j-2 ){
     for(i = 1; i <= nx - 1; i=i+2 ){
    printf("%c",a[i][j]);
   }
    printf("\n");
  }

  } //loop1

  fp = fopen("dataphi","w");
    for(j = 0; j <= ny; j++){ for(i = 0; i <= nx; i++){
      fprintf(fp," %10.8e", phi[i][j]);
    }
  fprintf(fp,"\n");
  } 
  fclose(fp);

  return 0;
}
