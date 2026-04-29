C*********************************************************************
C*********************************************************************
C*                                                                   *
C*        格子 Boltzmann 法による２次元非圧縮 Couette 流れ           *
C*                                                                   *
C*                                                                   *
C*                                              28th, Jan., 1998     *
C*                                                                   *
C*                        Program Name        : lbcoute.for          *
C*                        File Name for Input : lbcoute.dat          *
C*                                                                   *
C*                        programed by       Naoki TAKADA            *
C*                                           Masahiro YOSHIDA        *
C*                   ----------------------------------------------  *
C*                   Graduate School of Science and Technology       *
C*                   Kobe University                                 *
C*                   Rokkodai-cho 1-1, Nada, Kobe 657-8501, JAPAN    *
C*                                                                   *
C*********************************************************************
C*********************************************************************
C---------------------------------------------------------------------
C      F(I,J,K): 分布関数(>0,負になると発散)
C               I,J: x,y方向の格子点番号(左側壁I=0,下方壁J=0)
C               (注:I=-1は境界条件のために左側壁内に仮想的に設けられた
C                   x方向格子点番号)
C               K  : 粒子の運動方向=0,1,...,6, (0は静止粒子)
C      U(I,J),V(I,J),P(I,J): x,y方向の流速成分および圧力
C      DMF(I,J,K): 並進前の粒子分布を記憶しておくための配列変数
C      DENS : 初期状態での運動粒子の平均密度
C      WU   : 上方・下方壁の移動速度(注:運動粒子の速さより十分小)
C      RE   : Reynolds数(代表長さは平板間距離)
C      M,N  : x,y方向の格子点数(注:Nは奇数であること)
C            (左側流入I=0,右側流出I=M-1,下方壁J=0,上方壁J=N-1)
C      TEN  : 終了タイムステップ
C      TIN  : データ出力のタイムステップ間隔
C      TAU  : 時間刻み
C      PI   : 円周率
C      SF0  : 定数=6,SIGPとSIGCの比率に相当
C      C    : 運動粒子の速さ(格子一辺の長さを1としてC*TAU=1.0)
C      SIGC : 静止粒子-運動粒子の衝突(2種類の運動粒子が生成)
C             のパラメータ(本文中の衝突断面積σ)
C      SIGP : 120°方向が異なる運動粒子の2体衝突
C             (静止粒子と運動粒子が生成)のパラメータ
C      SIG3 : 運動粒子の3体衝突(数値安定性のために導入)のパラメータ
C---------------------------------------------------------------------
       REAL F(0:20,0:200,0:6)
     &     ,U(0:20,0:200),V(0:20,0:200),P(0:20,0:200)
     &     ,FF(0:20,0:200,0:6),D,SIGP,SIGC,SIG3,VAL,SF0,PI,C,TAU,WU
       INTEGER M,N,TT,TI
       SF0=6.0E0
       PI=ATAN(1.0E0)*4.0E0

C
       CALL INPUT(D,TAU,C,VAL,M,N,TT,TI,WU,RE,PI)
C
       CALL INIT(D,M,N,F,SF0,C,SIGP,SIGC,SIG3,VAL)
C
       DO 10 I=1,TT
C
         WRITE(*,*) 'TIMESTEP=',I
C
         CALL COLLI2(F,FF,M,N,SIGP,SIGC,SIG3,WU,SF0,C,PI)
C
         CALL MOVE(F,M,N)
C
         CALL BC(F,FF,M,N,WU,SF0,C,PI)
C
         IF((I.GE.TI).AND.(MOD(I,TI).EQ.0)) THEN
C
           CALL CALC(F,U,V,P,M,N,PI,SF0,C)
C
           CALL AVERAGE(M,N,U,V,P,D,TAU,C,RE,WU,TT,TI,I)
C
         ENDIF
C
   10  CONTINUE
C
       STOP
       END
C
C
C*********************************************************************
C*     パラメータの入力　                                            *
C*********************************************************************
       SUBROUTINE INPUT(D,TAU,C,VAL,M,N,TT,TI,WU,RE,PI)
C---------------------------------------------------------------------
       REAL D,TAU,C,VAL,WU,RE,PI
       INTEGER M,N,TT,TI
C
       OPEN(8,FILE='lbcoute.dat',STATUS='OLD')
       READ(8,*) M,N,TT,TI
       READ(8,*) D,TAU,C,RE
       READ(8,*) WU
       CLOSE(8)
C------[ Special Condition ]
       WU=WU*SIN(PI/3.0E0)
C---------------------------
       VAL=1.0E0/(8.0E0*REAL(N-2)*SIN(PI/3.0)*WU/TAU/C/C/RE+1.0E0)
C
       RETURN
       END
C
C
C*********************************************************************
C*  　 初期値の入力（静止状態） 　　　　　　　　　　　               *
C*********************************************************************
       SUBROUTINE INIT(D,M,N,F,SF0,C,SIGP,SIGC,SIG3,VAL)
C---------------------------------------------------------------------
       REAL F(0:20,0:200,0:6),SF0,C,SIGP,SIGC,SIG3,VAL,ROU
C
       ROU=D*(SF0+6.0E0)
       SIGP=0.5E0*VAL/D
       SIGC=SIGP/6.0E0
       SIG3=(1.0E0-9.0E0*D*SIGP)/6.0E0/D/D
C
       DO 20 I=0,M-1
         DO 30 J=0,N-1
           DO 35 K=0,6
             F(I,J,K)=EQUIV(0.0E0,0.0E0,0.0E0,PI,SF0,C,ROU,K)
   35      CONTINUE
   30    CONTINUE
   20  CONTINUE
C
       RETURN
       END
C
C
C*********************************************************************
C*     粒子の衝突演算　　　　　　　　　　　　                        *
C*********************************************************************
       SUBROUTINE COLLI2(F,FF,M,N,SIGP,SIGC,SIG3,WU,SF0,C,PI)
C---------------------------------------------------------------------
       REAL F(0:20,0:200,0:6),FF(0:20,0:200,0:6),SIGP,SIGC,SIG3
     &     ,F0,F1,F2,F3,F4,F5,F6
C
       DO 40 I=0,M-1
         DO 50 J=0,N-1
           DO 60 K=0,6
             FF(I,J,K)=F(I,J,K)
   60      CONTINUE
   50    CONTINUE
   40  CONTINUE
C
       DO 70 I=0,M-1
         DO 80 J=1,N-2
C
           F0=F(I,J,0)
           F1=F(I,J,1)
           F2=F(I,J,2)
           F3=F(I,J,3)
           F4=F(I,J,4)
           F5=F(I,J,5)
           F6=F(I,J,6)
           F(I,J,0)=F0+SIGP*(F1*F3+F2*F4+F3*F5+F4*F6+F5*F1+F6*F2)
     &                -SIGC*F0*(F1+F2+F3+F4+F5+F6)
           F(I,J,1)=F1+SIGC*F0*(F2+F6-F1)-SIGP*(F1*(F3+F5)-F2*F6)
     &                +SIG3*(F2*F4*F6-F1*F3*F5)
           F(I,J,2)=F2+SIGC*F0*(F1+F3-F2)-SIGP*(F2*(F4+F6)-F1*F3)
     &                +SIG3*(F1*F3*F5-F2*F4*F6)
           F(I,J,3)=F3+SIGC*F0*(F2+F4-F3)-SIGP*(F3*(F1+F5)-F2*F4)
     &                +SIG3*(F2*F4*F6-F1*F3*F5)
           F(I,J,4)=F4+SIGC*F0*(F3+F5-F4)-SIGP*(F4*(F6+F2)-F3*F5)
     &                +SIG3*(F1*F3*F5-F2*F4*F6)
           F(I,J,5)=F5+SIGC*F0*(F4+F6-F5)-SIGP*(F5*(F1+F3)-F4*F6)
     &                +SIG3*(F2*F4*F6-F1*F3*F5)
           F(I,J,6)=F6+SIGC*F0*(F5+F1-F6)-SIGP*(F6*(F4+F2)-F5*F1)
     &                +SIG3*(F1*F3*F5-F2*F4*F6)
C
   80    CONTINUE
   70  CONTINUE
C
       DO 90 I=1,M-2
         DO 100 J=1,N-2
C
           F0=FF(I,J,0)
           F1=FF(I,J,1)
           F2=FF(I,J,2)
           F3=FF(I,J,3)
           F4=FF(I,J,4)
           F5=FF(I,J,5)
           F6=FF(I,J,6)
           DM0=F(I,J,0)
           DM1=F(I-1,J,1)
           DM4=F(I+1,J,4)
           IF(MOD(J,2).EQ.1) THEN
             DM2=F(I,J-1,2)
             DM3=F(I+1,J-1,3)
             DM6=F(I,J+1,6)
             DM5=F(I+1,J+1,5)
           ELSE
             DM2=F(I-1,J-1,2)
             DM3=F(I,J-1,3)
             DM6=F(I-1,J+1,6)
             DM5=F(I,J+1,5)
           ENDIF
           FF(I,J,0)=F0+0.5E0*SIGP*( F1*F3+F2*F4+F3*F5
     &                              +F4*F6+F5*F1+F6*F2 )
     &                 -0.5E0*SIGC*F0*(F1+F2+F3+F4+F5+F6)
     &                 +0.5E0*SIGP*( DM1*DM3+DM2*DM4+DM3*DM5
     &                              +DM4*DM6+DM5*DM1+DM6*DM2 )
     &                 -0.5E0*SIGC*DM0*(DM1+DM2+DM3+DM4+DM5+DM6)
           FF(I,J,1)=F1+0.5E0*( SIGC*F0*(F2+F6-F1)
     &                         -SIGP*(F1*(F3+F5)-F2*F6) )
     &                 +0.5E0*SIG3*(F2*F4*F6-F1*F3*F5)
     &                 +0.5E0*( SIGC*DM0*(DM2+DM6-DM1)
     &                         -SIGP*(DM1*(DM3+DM5)-DM2*DM6) )
     &                 +0.5E0*SIG3*(DM2*DM4*DM6-DM1*DM3*DM5)
           FF(I,J,2)=F2+0.5E0*( SIGC*F0*(F1+F3-F2)
     &                         -SIGP*(F2*(F4+F6)-F1*F3) )
     &                  +0.5E0*SIG3*(F1*F3*F5-F2*F4*F6)
     &                  +0.5E0*( SIGC*DM0*(DM1+DM3-DM2)
     &                          -SIGP*(DM2*(DM4+DM6)-DM1*DM3) )
     &                  +0.5E0*SIG3*(DM1*DM3*DM5-DM2*DM4*DM6)
           FF(I,J,3)=F3+0.5E0*( SIGC*F0*(F2+F4-F3)
     &                         -SIGP*(F3*(F1+F5)-F2*F4) )
     &                 +0.5E0*SIG3*(F2*F4*F6-F1*F3*F5)
     &                 +0.5E0*( SIGC*DM0*(DM2+DM4-DM3)
     &                         -SIGP*(DM3*(DM1+DM5)-DM2*DM4) )
     &                 +0.5E0*SIG3*(DM2*DM4*DM6-DM1*DM3*DM5)
           FF(I,J,4)=F4+0.5E0*( SIGC*F0*(F3+F5-F4)
     &                         -SIGP*(F4*(F6+F2)-F3*F5) )
     &                 +0.5E0*SIG3*(F1*F3*F5-F2*F4*F6)
     &                 +0.5E0*( SIGC*DM0*(DM3+DM5-DM4)
     &                         -SIGP*(DM4*(DM6+DM2)-DM3*DM5) )
     &                 +0.5E0*SIG3*(DM1*DM3*DM5-DM2*DM4*DM6)
           FF(I,J,5)=F5+0.5E0*( SIGC*F0*(F4+F6-F5)
     &                         -SIGP*(F5*(F1+F3)-F4*F6) )
     &                 +0.5E0*SIG3*(F2*F4*F6-F1*F3*F5)
     &                 +0.5E0*( SIGC*DM0*(DM4+DM6-DM5)
     &                         -SIGP*(DM5*(DM1+DM3)-DM4*DM6) )
     &                 +0.5E0*SIG3*(DM2*DM4*DM6-DM1*DM3*DM5)
           FF(I,J,6)=F6+0.5E0*( SIGC*F0*(F5+F1-F6)
     &                         -SIGP*(F6*(F4+F2)-F5*F1) )
     &                 +0.5E0*SIG3*(F1*F3*F5-F2*F4*F6)
     &                 +0.5E0*( SIGC*DM0*(DM5+DM1-DM6)
     &                         -SIGP*(DM6*(DM4+DM2)-DM5*DM1) )
     &                 +0.5E0*SIG3*(DM1*DM3*DM5-DM2*DM4*DM6)
C
  100    CONTINUE
   90  CONTINUE
C
       CALL PREBC1(F,FF,M,N,WU,SF0,C,PI)
C
       CALL PREBC2(FF,M,N)
C
       TOTAL=0.0E0
       DO 110 I=0,M-1
         DO 120 J=0,N-1
           DO 130 K=0,6
             TOTAL=TOTAL+FF(I,J,K)
             F(I,J,K)=FF(I,J,K)
  130      CONTINUE
  120    CONTINUE
  110  CONTINUE
       WRITE(*,*) 'Total of Number of Particles =',TOTAL
C
       RETURN
       END
C
C
C*********************************************************************
C*     粒子の並進演算　　　　　                                      *
C*********************************************************************
       SUBROUTINE MOVE(F,M,N)
C---------------------------------------------------------------------
       REAL F(0:20,0:200,0:6)
C-------------------------------------------[1-DIRECTION]
       DO 1010 I=M-1,1,-1
         DO 1000 J=N-2,1,-1
           F(I,J,1)=F(I-1,J,1)
 1000    CONTINUE
 1010  CONTINUE
C-------------------------------------------[4-DIRECTION]
       DO 1030 I=0,M-2
         DO 1020 J=1,N-2
           F(I,J,4)=F(I+1,J,4)
 1020    CONTINUE
 1030  CONTINUE
C-------------------------------------------[2-Direction]
       DO 1050 J=N-1,2,-2
         DO 1040 I=M-1,1,-1
           F(I,J,2)=F(I-1,J-1,2)
           F(I,J-1,2)=F(I,J-2,2)
 1040    CONTINUE
         F(0,J-1,2)=F(0,J-2,2)
 1050  CONTINUE
C-------------------------------------------[3-Direction]
       DO 1070 J=N-1,2,-2
         DO 1060 I=0,M-2
           F(I,J,3)=F(I,J-1,3)
           F(I,J-1,3)=F(I+1,J-2,3)
 1060    CONTINUE
         F(M-1,J,3)=F(M-1,J-1,3)
 1070  CONTINUE
C-------------------------------------------[5-Direction]
       DO 1090 J=0,N-3,2
         DO 1080 I=0,M-2
           F(I,J,5)=F(I,J+1,5)
           F(I,J+1,5)=F(I+1,J+2,5)
 1080    CONTINUE
         F(M-1,J,5)=F(M-1,J+1,5)
 1090  CONTINUE
C-------------------------------------------[6-Direction]
       DO 1110 J=0,N-3,2
         DO 1100 I=M-1,1,-1
           F(I,J,6)=F(I-1,J+1,6)
           F(I,J+1,6)=F(I,J+2,6)
 1100    CONTINUE
         F(0,J+1,6)=F(0,J+2,6)
 1110  CONTINUE
C
       RETURN
       END
C
C
C*********************************************************************
C*     壁面での境界条件と周期境界条件　　　　                        *
C*********************************************************************
       SUBROUTINE PREBC1(F,FF,M,N,WU,SF0,C,PI)
C---------------------------------------------------------------------
       REAL F(0:20,0:200,0:6),FF(0:20,0:200,0:6)
C
       DO 245 I=1,M-2
         ROU=FF(I,0,0)+FF(I,0,1)+FF(I,0,4)+FF(I,0,5)+FF(I,0,6)
     &      +F(I,1,5)+F(I-1,1,6)
         DO 255 K=0,6
           FF(I,0,K)=0.5E0*EQUIV(0.0E0,0.0E0,0.0E0,PI,SF0,C,ROU,K)
     &              +0.5E0*F(I,0,K)
  255    CONTINUE
  245  CONTINUE
C
       DO 265 I=1,M-2
         ROU=FF(I,N-1,0)+FF(I,N-1,1)+FF(I,N-1,2)+FF(I,N-1,3)
     &      +FF(I,N-1,4)+F(I-1,N-2,2)+F(I,N-2,3)
         DO 275 K=0,6
           FF(I,N-1,K)=0.5E0*EQUIV(WU,0.0E0,WU*WU,PI,SF0,C,ROU,K)
     &                +0.5E0*F(I,N-1,K)
  275    CONTINUE
  265  CONTINUE
C
       RETURN
       END
C
C
C*********************************************************************
C*     壁面での境界条件と周期境界条件２　　　                        *
C*********************************************************************

       SUBROUTINE PREBC2(FF,M,N)
C---------------------------------------------------------------------
       REAL FF(0:20,0:200,0:6)
C
       DO 200 J=0,N-1
         DO 210 K=0,6
           FF(0,J,K)=FF(1,J,K)
           FF(M-1,J,K)=FF(M-2,J,K)
  210    CONTINUE
  200  CONTINUE
C
C
       RETURN
       END
C
C
C*********************************************************************
C*     壁面での境界条件と周期境界条件３　　　                        *
C*********************************************************************
       SUBROUTINE BC(F,FF,M,N,WU,SF0,C,PI)
C---------------------------------------------------------------------
       REAL F(0:20,0:200,0:6),FF(0:20,0:200,0:6),WU,SF0,C,PI,ROU
C
       DO 240 I=1,M-2
         ROU=FF(I,0,0)+FF(I,0,1)+FF(I,0,4)+FF(I,0,5)+FF(I,0,6)
     &      +F(I,0,5)+F(I,0,6)
         DO 250 K=0,6
           F(I,0,K)=EQUIV(0.0E0,0.0E0,0.0E0,PI,SF0,C,ROU,K)
  250    CONTINUE
  240  CONTINUE
C
       DO 260 I=1,M-2
         ROU=FF(I,N-1,0)+FF(I,N-1,1)+FF(I,N-1,2)+FF(I,N-1,3)
     &      +FF(I,N-1,4)+F(I,N-1,2)+F(I,N-1,3)
         DO 270 K=0,6
           F(I,N-1,K)=EQUIV(WU,0.0E0,WU*WU,PI,SF0,C,ROU,K)
  270    CONTINUE
  260  CONTINUE
C
       DO 220 J=0,N-1
         DO 230 K=0,6
           F(0,J,K)=F(1,J,K)
           F(M-1,J,K)=F(M-2,J,K)
  230    CONTINUE
  220  CONTINUE
C
       RETURN
       END
C
C
C*********************************************************************
C*     局所平衡分布関数　　　　　　　　　　　                        *
C*********************************************************************
       REAL FUNCTION EQUIV(U,V,U2,PI,SF0,C,ROU,K)
C---------------------------------------------------------------------
       REAL U,V,U2,PI,SF0,C,ROU
       IF(K.EQ.0) THEN
         EQUIV=0.5E0*ROU-ROU*U2/(C*C)
       ELSE
         CONST=C*(COS(PI*(K-1)/3.0E0)*U+SIN(PI*(K-1)/3.0E0)*V)
         EQUIV=ROU/(SF0+6.0E0)
     &         +ROU*CONST/(3.0E0*C*C)
     &         +2.0E0*ROU*(CONST*CONST-0.5E0*U2)/(3.0E0*C*C*C*C)
     &         +ROU*U2/(6.0E0*C*C)
       ENDIF
C
       RETURN
       END
C---------------------------------------------------------------------
C
C
C*********************************************************************
C*     マクロな変数の計算 (速度 & 圧力)                              *
C*********************************************************************
       SUBROUTINE CALC(F,U,V,P,M,N,PI,SF0,C)
C---------------------------------------------------------------------
       REAL F(0:20,0:200,0:6)
     &     ,U(0:20,0:200),V(0:20,0:200),P(0:20,0:200),SF0,C,PI,R
C
       DO 100 I=0,M-1
         DO 110 J=0,N-1
           R=0.0E0
           DO 120 K=0,6
             R=R+F(I,J,K)
  120      CONTINUE
           U(I,J)=C/R*(F(I,J,1)-F(I,J,4)+0.5*(F(I,J,2)+F(I,J,6)
     &                -F(I,J,3)-F(I,J,5)))
           V(I,J)=C/R*(COS(PI/6)*(F(I,J,2)+F(I,J,3)-F(I,J,5)
     &               -F(I,J,6)))
           P(I,J)=3.0E0*R*C*C/(6.0E0+SF0)*(1.0E0+(6.0E0+SF0)/6.0E0
     &              *(SF0-6.0E0)/12*(U(I,J)*U(I,J)+V(I,J)*V(I,J))/C/C)
  110    CONTINUE
  100  CONTINUE
C
       RETURN
       END
C
C
C*********************************************************************
C*     流れ場のデータの出力　　　　　                                *
C*********************************************************************
       SUBROUTINE OUTPUT(M,N,U,V,P,D,TAU,C,RE,WU,TT,TI,II)
C---------------------------------------------------------------------
       CHARACTER NAME*20
       INTEGER TT,TI
       REAL U(0:20,0:200),V(0:20,0:200),P(0:20,0:200)
     &     ,D,TAU,C,RE,WU
C
       K13=II
       K12=K13/1000000000
       K11=K13-K12*1000000000
       K10=K11/100000000
       KK9=K11-K10*100000000
       KK8=KK9/10000000
       KK7=KK9-KK8*10000000
       KK6=KK7/1000000
       KK5=KK7-KK6*1000000
       KK4=KK5/100000
       KK3=KK5-KK4*100000
       KK2=KK3/10000
       KK1=KK3-KK2*10000
       KK0=KK1/1000
       KKK=KK1-KK0*1000
       NAME='co-rk'//CHAR(KK4+48)//CHAR(KK2+48)//CHAR(KK0+48)//'.dat'
       OPEN(7,FILE=NAME,STATUS='UNKNOWN')
       WRITE(7,*) M,N,TT,TI
       WRITE(7,*) D,TAU,C,RE
       WRITE(7,*) WU
       DO 120 I=0,M-1
         DO 130 J=0,N-1
           WRITE(7,*) U(I,J),V(I,J),P(I,J)
  130    CONTINUE
  120  CONTINUE
       CLOSE(7)
C
       RETURN
       END
C
C
C*********************************************************************
C*     流れ場の平均されたデータの出力　　　　                        *
C*********************************************************************
       SUBROUTINE AVERAGE(M,N,U,V,P,D,TAU,C,RE,WU,TT,TI,II)
C---------------------------------------------------------------------
       CHARACTER NAME*20
       INTEGER TT,TI
       REAL U(0:20,0:200),V(0:20,0:200),P(0:20,0:200)
     &     ,D,TAU,C,RE,WU
C
       TIME=FLOAT(II)+0.001
       KK=INT(TIME)
       K13=KK
       K12=K13/10000000
       K11=K13-K12*10000000
       K10=K11/1000000
       KK9=K11-K10*1000000
       KK8=KK9/100000
       KK7=KK9-KK8*100000
       KK6=KK7/10000
       KK5=KK7-KK6*10000
       KK4=KK5/1000
       KK3=KK5-KK4*1000
       KK2=KK3/100
       KK1=KK3-KK2*100
       KK0=KK1/10
       KKK=KK1-KK0*10
       NAME='co-rk'//CHAR(KK8+48)//CHAR(KK6+48)//CHAR(KK4+48)//'.av1'
       OPEN(7,FILE=NAME,STATUS='UNKNOWN')
       WRITE(7,*) M,N
       WRITE(7,*) D,TAU,C,RE
       WRITE(7,*) TT,TI,II
       WRITE(7,*) WU
       DO 120 J=0,N-1
         AVU=0.0E0
         AVP=0.0E0
         DO 130 I=0,M-1
           AVU=AVU+U(I,J)/REAL(M)
           AVP=AVP+P(I,J)/REAL(M)
  130    CONTINUE
         IF(J.EQ.0) THEN
           WRITE(7,*) J,REAL(J)*TAU*C,AVU,AVP
         ELSE IF(J.EQ.N-1) THEN
           WRITE(7,*) J,REAL(N-2)*TAU*C,AVU,AVP
         ELSE
           WRITE(7,*) J,(REAL(J)-0.5E0)*TAU*C,AVU,AVP
         ENDIF
  120  CONTINUE
       CLOSE(7)
C
       RETURN
       END
