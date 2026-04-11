// IB-LBM  
// Immersed Boundary-Lattice Boltzmann Method.
// The source code is written in C programming language.
// We need a workstation to run this program. We must reduce mesh size in Cygwin.
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
// nx : number of grid points (x-axis)
// ny : number of grid points (y-axis)
// f  : distribution function
// f0 : equilibrium distribution function
// cx : discrete velocity (x-axis)
// cy : discrete velocity (y-axis)
// tau: relaxation time
// nu : kinematic viscosity
// gx : gravity (x-axis)
// gy : gravity (y-axis)
// dt : time step
//
// np  : number of particles
// rp  : radius of particle
// rhop: density of particle
// mp  : particle mass divided by density
// iip : particle inertia divided by density 
// xp  : position of particles (x-axis, time = n)
// yp  : position of particles (y-axis, time = n)
// xp1 : position of particles (x-axis, time = n)
// yp1 : position of particles (y-axis, time = n - 1)
// fxp : force acting on particles (x-axis)
// fyp : force acting on particles (y-axis)
// up  : horizontal velocity component  of particles (time = n)
// vp  : vertical velocity component   of particles (time = n)
// up1 : horizontal velocity component  of particles (time = n - 1)
// vp1 : vertical velocity component   of particles (time = n - 2)
// up2 : horizontal velocity component  of particles (time = n - 1)
// vp2 : vertical velocity component   of particles (time = n - 2)
// fxc : collision  force acting on particles (x-axis)
// fyc : collision  force acting on particles (y-axis) 
// theta : angle of particles ( time = n)
// torq  : torque acting on particles
// omega : angular velocity of particles (time = n)
// omega1: angular velocity of particles (time = n - 1)
// omega2: angular velocity of particles (time = n - 2)
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
// epsw: parameter for collision force
// zeta: parameter for collision force


#include <stdio.h>
#include <math.h>
#include <stdlib.h>

//# define DIM 150
# define DIM 500
# define ed 2
# define pd 200

int main(void)
{
  FILE   *fp;
  int    nx = 100, ny = 400, time = 0, loop1, loop2;
  int    i, j, k, n, m, in, jn, ip, im, jp, jm;
  double rho[DIM][DIM], u[DIM][DIM], v[DIM][DIM], un[DIM][DIM], vn[DIM][DIM];
  double fx[DIM][DIM], fy[DIM][DIM];
  double f[9][DIM][DIM], f0[9][DIM][DIM], ftmp[9][DIM][DIM], cx[9], cy[9];
  double epsw = 100.0, zeta =1.0;
  double umax, umin, tmp, u2, nu, tau;

  double xp[ed], yp[ed], xp1[ed], yp1[ed], fxp[ed], fyp[ed];
  double up[ed], vp[ed], up1[ed], vp1[ed], up2[ed], vp2[ed];
  double theta[ed], torq[ed], omega[ed], omega1[ed], omega2[ed];
  double fxc[ed], fyc[ed];

  double xe[ed][pd], ye[ed][pd], fxe[ed][pd], fye[ed][pd];
  double uet[ed][pd], vet[ed][pd], ue[ed][pd], ve[ed][pd];

  double tmp1, tmp2, tmp3, gx, gy, rhop, dt, rp, mp, iip;
  int    np, ne;
  char   a[DIM][DIM];

  // initial condition
  tau = 0.53;
  nu = (tau - 0.5)/3.0;

// particle
  np = 1;
  rp = 0.125*(double)nx/2.0;
  rhop= 1.25;
  ne = (int)(2.0 * M_PI * rp * 2.0);
  mp = M_PI*rp*rp;
  iip = M_PI*rp*rp*rp*rp*0.5;

  dt = nu/((double)nx/2.0)/((double)nx/2.0)/0.1;
  gy = -981.0/2.0*(double)nx*dt*dt;
  gx = 0.0;

  printf("np=%d, ne=%d, rp=%f, dt=%f, gy=%f, nu=%f\n", np, ne, rp, dt, gy, nu);

  for(n = 0; n < np; n++) {
       up[n] = 0.0;    up1[n] = 0.0;    up2[n] = 0.0;
       vp[n] = 0.0;    vp1[n] = 0.0;    vp2[n] = 0.0;
    omega[n] = 0.0; omega1[n] = 0.0; omega2[n] = 0.0;
      fxp[n] = 0.0;    fyp[n] = 0.0;   torq[n] = 0.0;  theta[n] = 0.0;
  }

  xp[0] = 1.0/2.0*(double)nx;
  yp[0] = 4.0/2.0*(double)nx;

  for(n = 0; n < np; n++) { for(m = 0; m <ne ; m++) {
     xe[n][m] = xp[n] + rp*cos(2.0*M_PI*(double)m/(double)ne);
     ye[n][m] = yp[n] + rp*sin(2.0*M_PI*(double)m/(double)ne);
    fxe[n][m] = 0.0; fye[n][m] = 0.0; ue[n][m] = 0.0; ve[n][m] = 0.0;
  } } 

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
  for(loop1 = 0; loop1 < 700; loop1++){
  for(loop2 = 0; loop2 < 40; loop2++){

    fp=fopen("yps","a+");
      fprintf(fp,"%f\n",yp[0]*2.0/(double)nx);
    fclose(fp);

    fp=fopen("vps","a+");
      fprintf(fp,"%f\n",vp[0]/dt/(double)nx*2);
    fclose(fp);

    fp=fopen("times","a+");
    fprintf(fp,"%f\n",time*dt);
    fclose(fp);


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

      f[4][i][ny-1] = f[2][i ][ny];
      f[7][i][ny-1] = f[5][ip][ny];
      f[8][i][ny-1] = f[6][im][ny];
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

// immersed boundary method
    for(n = 0; n < np; n++) { for(m = 0; m <ne ; m++) {
      uet[n][m] = 0.0; vet[n][m] = 0.0;
    } }

    for(n = 0; n < np; n++) { for(m = 0; m <ne ; m++) {
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

    for(n = 0; n < np; n++) { for(m = 0; m <ne ; m++) {
      fxe[n][m] = 0.0; fye[n][m] = 0.0; 
    } }

    for(n = 0; n < np; n++) { for(m = 0; m <ne ; m++) {
      fxe[n][m] = ue[n][m] - uet[n][m];
      fye[n][m] = ve[n][m] - vet[n][m];
    } }

// Spreading: force (fxpe, fype) --> (fx, fy)
    for(i = 0; i < nx; i++) { for(j = 0; j < ny; j++) {
      fx[i][j] = 0.0; fy[i][j] = 0.0;
    } }

    for(n = 0; n < np; n++) { for(m = 0; m <ne ; m++) {
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

        fx[i][j] += fxe[n][m] * tmp3 * 2.0f*M_PI*rp/(double)ne;
        fy[i][j] += fye[n][m] * tmp3 * 2.0f*M_PI*rp/(double)ne;
      } }
    } }

    for(n = 0; n < np; n++){
      fxc[n] = 0.0;
      fyc[n] = 0.0;
    }

// particle-wall collision
// bottom wall 
    tmp1 = fabs(yp[n] + rp); 
    if(tmp1 < 2.0*rp + zeta){
      fyc[n] += (yp[n] + rp)*(2.0*rp - tmp1 + zeta)*(2.0*rp - tmp1 + zeta)/epsw;
    } 

// particle motion
    for(n = 0; n < np; n++){
      fxp[n] = 0.0; fyp[n] = 0.0; torq[n] = 0.0;
    }
    for(n = 0; n < np; n++){ for(m = 0; m <ne ; m++){
       fxp[n] += fxe[n][m];
       fyp[n] += fye[n][m];
      torq[n]  = torq[n] + fye[n][m]*(xe[n][m] - xp[n])
                         - fxe[n][m]*(ye[n][m] - yp[n]);
    } }
    for(n = 0; n < np; n++){
       fxp[n] =  -fxp[n]*2.0f*M_PI*rp/(double)ne;
       fyp[n] =  -fyp[n]*2.0f*M_PI*rp/(double)ne;
      torq[n] = -torq[n]*2.0f*M_PI*rp/(double)ne;
    }

    for(n = 0; n < np; n++){
       up[n]  = (1.0 + 1.0/rhop)*up1[n]    - 1.0/rhop*up2[n]
              + (fxp[n] + fxc[n])/rhop/mp + (1.0 - 1.0/rhop)*gx;

       vp[n]  = (1.0 + 1.0/rhop)*vp1[n]    - 1.0/rhop*vp2[n]
              + (fyp[n] + fyc[n])/rhop/mp + (1.0 - 1.0/rhop)*gy;

      omega[n]= (1.0 + 1.0/rhop)*omega1[n] - 1.0/rhop*omega2[n]
              + torq[n]/rhop/iip;

         xp[n]=    xp[n] + (   up[n] +    up1[n])*0.5;
         yp[n]=    yp[n] + (   vp[n] +    vp1[n])*0.5;
      theta[n]= theta[n] + (omega[n] + omega1[n])*0.5;

         up2[n] =    up1[n];    up1[n] =    up[n];
         vp2[n] =    vp1[n];    vp1[n] =    vp[n];
      omega2[n] = omega1[n]; omega1[n] = omega[n]; 
    }

    for(n = 0; n < np; n++) { for(m = 0; m <ne ; m++) {
      xe[n][m] = xp[n] + rp*cos(2.0f*M_PI*m/(double)ne);
      ye[n][m] = yp[n] + rp*sin(2.0f*M_PI*m/(double)ne);
    } } 

    for(n = 0; n < np; n++) { for(m = 0; m <ne ; m++) {
      ue[n][m] = up[n]  - omega[n]*(ye[n][m] - yp[n]);
      ve[n][m] = vp[n]  + omega[n]*(xe[n][m] - xp[n]);
    } }

  } //loop2
  
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

  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){ for(n = 0; n < ne; n++){
    if(pow((double)i - xp[n],2) + pow((double)j - yp[n],2) <= pow(rp,2)){
      a[i][j] = '*';
    }
  } } }

  for(j = ny - 1; j >= 1; j = j - 5){ for(i = 1; i <= nx - 1; i = i + 5){
    printf("%c",a[i][j]);
    }
    printf("\n");
  }

    printf("%f\t,x:%f,y:%f,u:%f,v%f,theta:%f\n",time*dt,xp[0],yp[0]*2.0/(double)nx,up[0],vp[0]/dt/(double)nx*2,theta[0]);

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
