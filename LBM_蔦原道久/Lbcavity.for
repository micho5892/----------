C*********************************************************************
C*********************************************************************
C*                                                                   *
C*    　　　格子ボルツマン法による２次元キャビティー内の 　　        *
C*    　　　　　　　　　　　　非圧縮性流れ 　　　　　　　　　　　    *
C*                   　　　　　　　　　　　　　　                    *
C*    (with Collision Model Proposed by G.McNamara et al in 1992)    *
C*              (1-speed Model on Hexagonal Lattice)                 *
C*                                                                   *
C*                                    20/July/1994 (first version)   *
C*                                    30/Sep./1997 (partly revised)  *
C*                                                                   *
C*                                    Program Name    : lbcavity.for *
C*                                    Input Data File : lbcavity.dat *
C*                                                                   *
C*                        programmed by    NAOKI TAKADA              *
C*                        -----------------------------------------  *
C*                        Graduate School of Science and Technology  *
C*                        Kobe University                            *
C*                        Rokkodai-cho 1-1, Nada, Kobe 657, JAPAN    *
C*                                                                   *
C*********************************************************************
C*********************************************************************
C---------------------------------------------------------------------
C      ○キャビティ流れ Cavity Flow
C         最初四方を固体の壁で囲まれた流体は静止しているが,計算開始と
C         同時に上方の壁が一定速度で動き出すと粘性により流れが生じる.
C         時間が十分すぎると流れは定常状態に達する.
C                上方壁(移動)
C               ----->(一定速度)
C            +------------------+
C            |                  |
C            |                  |
C            |                  |
C      左側壁|       流体       |右側壁
C      (静止)|   (最初は静止)   |(静止)
C            |                  |
C            |                  |
C            +------------------+
C                下方壁(静止)
C---------------------------------------------------------------------
C      ○本プログラム内の変数
C
C      F(I,J,K): 分布関数(>0,負になると発散)
C               I,J: x,y方向の格子点番号(左側壁I=0,下方壁J=0)
C               (注:I=-1は境界条件のために左側壁内に仮想的に設けられた
C                   x方向格子点番号)
C               K  : 粒子の運動方向=0,1,...,6, (0は静止粒子)
C      U(I,J),V(I,J),P(I,J): x,y方向の流速成分および圧力
C      DMF(I,J,K): 並進前の粒子分布を記憶しておくための配列変数
C      DENS : 初期状態での運動粒子の平均密度
C      WALLV: 上方壁の移動速度(注:運動粒子の速さより十分小さい値)
C      RE   : Reynolds数(代表長さは移動壁の幅)
C      M,N  : x,y方向の格子点数(注:本プログラムでは偶数であること！)
C             (右側壁I=M-1,上方壁J=N-1)
C      TEN  : 終了タイムステップ
C      TIN  : データ出力のタイムステップ間隔
C      TAU  : 時間刻み
C      PI   : 円周率
C      LF0  : 定数=6,SIGPとSIGCの比率に相当
C      C    : 運動粒子の速さ(格子一辺の長さを1としてC*TAU=1.0)
C      SIGC : 静止粒子-運動粒子の衝突(2種類の運動粒子が生成)
C             のパラメータ(文献中の衝突断面積σ)
C      SIGP : 120°方向が異なる運動粒子の2体衝突
C             (静止粒子と運動粒子が生成)のパラメータ
C      SIG3 : 運動粒子の3体衝突(数値安定性のために導入)のパラメータ
C---------------------------------------------------------------------
       COMMON /VAR1/ F(-1:40,0:40,0:6)
       COMMON /VAR2/ U(0:40,0:40),V(0:40,0:40),P(0:40,0:40)
       COMMON /VAR3/ DMF(-1:40,0:40,0:6)
       COMMON /CON1/ DENS,WALLV,RE
       COMMON /CON2/ M,N
       COMMON /CON3/ TEN,TIN,TAU
       COMMON /CON4/ PI,LF0,C
       COMMON /CON5/ SIGC,SIGP,SIG3
C
       REAL F,U,V,P,DMF,DENS,WALLV,RE,TAU,PI,LF0,C,SIGC,SIGP,SIG3
       INTEGER TEN,TIN
       PI=ATAN(1.0E0)*4.0E0
C
C      [ データ入力(ファイルの読み込み) ]
       CALL INPUT
C
       WRITE(*,*) '<<<<<  CALCULATION START!!!  >>>>>'
C
C      [ 変数の初期化 ]
       CALL INIT
C
       DO 10 II=1,TEN
C
         WRITE(*,*) 'TIME STEP = ',II
C
C        [ 衝突演算 ]
         CALL COLLI
C
C        [ 並進 ]
         CALL MOVE
C
C        [ 境界条件 ]
         CALL BOUND
C
         IF((II.GE.TIN).AND.(MOD(II,TIN).EQ.0)) THEN
C
C        [ 流速・圧力の計算 ]
           CALL CALC
C
C        [ 出力 ]
           CALL OUTPUT(II)
C
         ENDIF
C
   10  CONTINUE
C
       STOP
       END
C
C
C
C*********************************************************************
C*     流れ場のパラメータ入力　　　                                  *
C*********************************************************************
       SUBROUTINE INPUT
C---------------------------------------------------------------------
       COMMON /CON1/ DENS,WALLV,RE
       COMMON /CON2/ M,N
       COMMON /CON3/ TEN,TIN,TAU
       COMMON /CON4/ PI,LF0,C
C
       REAL DENS,WALLV,RE,TAU,PI,LF0,C
       INTEGER TEN,TIN
C
       WRITE(*,*) '<< SUBROUTINE INPUT >>'
C
       OPEN(8,FILE='lbcavity.dat',STATUS='OLD')
       READ(8,*) DENS,WALLV,RE
       READ(8,*) M,N
       READ(8,*) TEN,TIN
       READ(8,*) LF0,TAU,C
       CLOSE(8)
C
       RETURN
       END
C
C
C********************************************************************
C*     粒子密度関数の初期化　　　　　　　　　　　　　　             *
C********************************************************************
       SUBROUTINE INIT
C--------------------------------------------------------------------
       COMMON /VAR1/ F(-1:40,0:40,0:6)
       COMMON /CON1/ DENS,WALLV,RE
       COMMON /CON2/ M,N
       COMMON /CON3/ TEN,TIN,TAU
       COMMON /CON4/ PI,LF0,C
       COMMON /CON5/ SIGC,SIGP,SIG3
C
       REAL F,DENS,WALLV,RE,TAU,PI,LF0,C,ROU,XNYU,VAL,SIGC,SIGP,SIG3
       INTEGER TEN,TIN
C
C      [ 初期の流体の平均密度　]
       ROU=(LF0+6.0E0)*DENS
C      [ 動粘性係数 ]
       XNYU=FLOAT(M-1)*WALLV/RE
C
       VAL=1.0E0/(8.0E0*XNYU/TAU/C/C+1.0E0)
C---------------------------------------------------------------------
C      LBM での動粘性係数=C*C*TAU*(1.0E0/VAL-1.0E0)/8.0E0
C        (VAL=2*SIGP*DENS)
C      レイノルズ数=FLOAT(M-1)*WALLV/XNYU
C---------------------------------------------------------------------
       SIGP=0.5E0*VAL/DENS
       SIGC=SIGP/LF0
       SIG3=(1.0E0-4.5E0*VAL)/ROU/ROU*(LF0/6.0E0+1.0E0)*(LF0+
     &      6.0E0)
C
       DO 80 I=-1,M
         DO 70 J=0,N-1
           DO 60 K=0,6
             F(I,J,K)=EQUIV(0.0E0,0.0E0,0.0E0,PI,LF0,C,ROU,K)
   60      CONTINUE
   70    CONTINUE
   80  CONTINUE
C
       RETURN
       END
C
C
C*********************************************************************
C*     衝突過程 (Multi-Particle Collision Model) 　　　　　          *
C*********************************************************************
       SUBROUTINE COLLI
C---------------------------------------------------------------------
       COMMON /VAR1/ F(-1:40,0:40,0:6)
       COMMON /VAR3/ DMF(-1:40,0:40,0:6)
       COMMON /CON2/ M,N
       COMMON /CON5/ SIGC,SIGP,SIG3
C
       REAL F,DMF,F0,F1,F2,F3,F4,F5,F6,SIGP,SIGC,SIG3
C
       DO 110 I=0,M-1
         DO 100 J=1,N-2
C
C          [ 側面での剛体壁における例外措置 ]
           IF(((I.EQ.0).AND.(MOD(J,2).EQ.1)).OR.
     &        ((I.EQ.M-1).AND.(MOD(J,2).EQ.0))) GOTO 100
C
           F0=F(I,J,0)
           F1=F(I,J,1)
           F2=F(I,J,2)
           F3=F(I,J,3)
           F4=F(I,J,4)
           F5=F(I,J,5)
           F6=F(I,J,6)
C
           F(I,J,0)=F0+SIGP*(F1*F3+F2*F4+F3*F5+F4*F6+F5*F1+F6*F2)
     &	              -SIGC*F0*(F1+F2+F3+F4+F5+F6)
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
  100    CONTINUE
  110  CONTINUE
C
C      [ 粒子数の確認 ]
       TOTAL=0.0E0
       DO 118 I=-1,M
         DO 117 J=0,N-1
           DO 116 K=0,6
             TOTAL=TOTAL+F(I,J,K)
             DMF(I,J,K)=F(I,J,K)
  116      CONTINUE
  117    CONTINUE
  118  CONTINUE
       WRITE(*,*) 'Total of Particle Density=',TOTAL
C
       RETURN
       END
C
C
C*********************************************************************
C*     粒子の並進過程　　　　　                                      *
C*********************************************************************
       SUBROUTINE MOVE
C---------------------------------------------------------------------
       COMMON /VAR1/ F(-1:40,0:40,0:6)
       COMMON /CON2/ M,N
C
       REAL F
C
C-------------------------------------------[1-DIRECTION]
       DO 130 J=1,N-2
         DO 120 I=M,0,-1
           IF((I.EQ.M).AND.(MOD(J,2).EQ.0)) GOTO 120
           IF((I.EQ.0).AND.(MOD(J,2).EQ.1)) GOTO 120
           F(I,J,1)=F(I-1,J,1)
  120    CONTINUE
  130  CONTINUE
C-------------------------------------------[2-DIRECTION]
       DO 150 J=N-1,2,-2
         DO 140 I=M-1,1,-1
           F(I,J,2)=F(I-1,J-1,2)
           F(I,J-1,2)=F(I,J-2,2)
  140    CONTINUE
         F(0,J-1,2)=F(0,J-2,2)
  150  CONTINUE
       DO 155 I=1,M-1
         F(I,1,2)=F(I-1,0,2)
  155  CONTINUE
C-------------------------------------------[3-DIRECTION]
       DO 170 J=N-1,2,-2
         DO 160 I=0,M-2
           F(I,J,3)=F(I,J-1,3)
           F(I,J-1,3)=F(I+1,J-2,3)
  160    CONTINUE
         F(M-1,J,3)=F(M-1,J-1,3)
  170  CONTINUE
       DO 175 I=0,M-1
         F(I,1,3)=F(I,0,3)
  175  CONTINUE
C-------------------------------------------[4-DIRECTION]
       DO 190 J=1,N-2
         DO 180 I=-1,M-1
           IF((I.EQ.-1).AND.(MOD(J,2).EQ.1)) GOTO 180
           IF((I.EQ.M-1).AND.(MOD(J,2).EQ.0)) GOTO 180
           F(I,J,4)=F(I+1,J,4)
  180    CONTINUE
  190  CONTINUE
C-------------------------------------------[5-DIRECTION]
       DO 210 J=0,N-3,2
         DO 200 I=0,M-2
           F(I,J,5)=F(I+1,J+1,5)
           F(I,J+1,5)=F(I,J+2,5)
  200    CONTINUE
         F(M-1,J+1,5)=F(M-1,J+2,5)
  210  CONTINUE
       DO 212 I=0,M-2
         F(I,N-2,5)=F(I+1,N-1,5)
  212  CONTINUE
C-------------------------------------------[6-DIRECTION]
       DO 220 J=0,N-3,2
         DO 215 I=M-1,1,-1
           F(I,J,6)=F(I,J+1,6)
           F(I,J+1,6)=F(I-1,J+2,6)
  215    CONTINUE
         F(0,J,6)=F(0,J+1,6)
  220  CONTINUE
       DO 222 I=0,M-1
         F(I,N-2,6)=F(I,N-1,6)
  222  CONTINUE
C
       RETURN
       END
C
C*********************************************************************
C*     剛体壁での境界条件　　　　　　　　　                          *
C*********************************************************************
       SUBROUTINE BOUND
C---------------------------------------------------------------------
       COMMON /VAR1/ F(-1:40,0:40,0:6)
       COMMON /VAR3/ DMF(-1:40,0:40,0:6)
       COMMON /CON1/ DENS,WALLV,RE
       COMMON /CON2/ M,N
       COMMON /CON3/ TEN,TIN,TAU
       COMMON /CON4/ PI,LF0,C
C
       INTEGER M,N,TEN,TIN
       REAL F,DMF,DENS,WALLV,RE,ROU,PI,LF0,C,TAU
C
C-------------------------------------------------------[TOP BOUMDARY]
       DO 260 I=1,M-1
         ROU=F(I,N-1,0)+F(I,N-1,1)+F(I,N-1,2)
     &      +F(I,N-1,3)+F(I,N-1,4)+DMF(I,N-1,2)+DMF(I,N-1,3)
         DO 250 K=0,6
           F(I,N-1,K)=EQUIV(WALLV,0.0E0,WALLV*WALLV,PI,LF0,C,ROU,K)
  250    CONTINUE
  260  CONTINUE
       F(0,N-1,6)=DMF(0,N-1,3)
C----------------------------------------------------[BOTTOM BOUNDARY]
       DO 270 I=0,M-2
         ROU=F(I,0,0)+F(I,0,1)+DMF(I,0,5)
     &      +DMF(I,0,6)+F(I,0,4)+F(I,0,5)+F(I,0,6)
         DO 265 K=0,6
           F(I,0,K)=EQUIV(0.0E0,0.0E0,0.0E0,PI,LF0,C,ROU,K)
  265    CONTINUE
  270  CONTINUE
       F(M-1,0,3)=DMF(M-1,0,6)
C------------------------------------------------[LEFT&RIGHT BOUNDARY]
       DO 280 J=1,N-2,2
         ROU=F(0,J,0)+DMF(0,J,4)+DMF(0,J,3)
     &      +DMF(0,J,5)+F(0,J,4)+F(0,J,3)+F(0,J,5)
         DO 275 K=0,6
           F(0,J,K)=EQUIV(0.0E0,0.0E0,0.0E0,PI,LF0,C,ROU,K)
  275    CONTINUE
         F(M,J,4)=DMF(M,J,1)
  280  CONTINUE
       DO 290 J=2,N-2,2
         ROU=F(M-1,J,0)+F(M-1,J,1)+F(M-1,J,2)
     &      +F(M-1,J,6)+DMF(M-1,J,1)+DMF(M-1,J,2)+DMF(M-1,J,6)
         DO 285 K=0,6
           F(M-1,J,K)=EQUIV(0.0E0,0.0E0,0.0E0,PI,LF0,C,ROU,K)
  285    CONTINUE
         F(-1,J,1)=DMF(-1,J,4)
  290  CONTINUE
C
       RETURN
       END
C
C*********************************************************************
C*     マクロな変数の計算 (流速 & 圧力)                              *
C*********************************************************************
       SUBROUTINE CALC
C---------------------------------------------------------------------
       COMMON /VAR1/ F(-1:40,0:40,0:6)
       COMMON /VAR2/ U(0:40,0:40),V(0:40,0:40),P(0:40,0:40)
       COMMON /CON2/ M,N
       COMMON /CON4/ PI,LF0,C
C
       REAL F,F0,F1,F2,F3,F4,F5,F6,LF0,C,PI,ROU,U2,U,V,P
C
       DO 310 I=0,M-1
         DO 300 J=0,N-1
C
           F0=F(I,J,0)
           F1=F(I,J,1)
           F2=F(I,J,2)
           F3=F(I,J,3)
           F4=F(I,J,4)
           F5=F(I,J,5)
           F6=F(I,J,6)
C
           ROU=F0+F1+F2+F3+F4+F5+F6
	   U(I,J)=(F1+COS(PI/3.0E0)*(F2+F6-F3-F5)-F4)*C/ROU
	   V(I,J)=(F2+F3-F5-F6)*C*SIN(PI/3.0E0)/ROU
           U2=U(I,J)*U(I,J)+V(I,J)*V(I,J)
           P(I,J)=3.0E0*ROU*C*C*(1.0E0+(LF0+6.0E0)*((LF0+6.0E0)
     &           /12.0E0-1.0E0)*U2/C/C/6.0E0)/(LF0+6.0E0)
C
  300    CONTINUE
  310  CONTINUE
C
       RETURN
       END
C
C
C*********************************************************************
C*     流れ場のデータの出力 　　　　　                               *
C*********************************************************************
       SUBROUTINE OUTPUT(II)
C---------------------------------------------------------------------
       COMMON /VAR2/ U(0:40,0:40),V(0:40,0:40),P(0:40,0:40)
       COMMON /CON1/ DENS,WALLV,RE
       COMMON /CON2/ M,N
       COMMON /CON3/ TEN,TIN,TAU
       COMMON /CON4/ PI,LF0,C
C
       REAL U,V,P,DENS,WALLV,RE,LF0,TAU,C
       INTEGER TEN,TIN
       CHARACTER NAME*25
C
       WRITE(*,*) '<< SUBROUTINE OUTPUT >>'
C   
C
       TIME=FLOAT(II)+0.001
       KK=INT(TIME)
       KK8=KK/100000
       KK7=KK-KK8*100000
       KK6=KK7/10000
       KK5=KK7-KK6*10000
       KK4=KK5/1000
       KK3=KK5-KK4*1000
       KK2=KK3/100
       KK1=KK3-KK2*100
       KK0=KK1/10
       KKK=KK1-KK0*10
       NAME='lbc'//CHAR(KK8+48)//CHAR(KK6+48)
     &           //CHAR(KK4+48)//CHAR(KK2+48)//CHAR(KK0+48)//'.dat'
C
       OPEN(9,FILE=NAME,STATUS='UNKNOWN')
       WRITE(9,*) DENS,WALLV,RE
       WRITE(9,*) M,N
       WRITE(9,*) TEN,TIN,II
       WRITE(9,*) LF0,TAU,C
C
C*****************************************************************
C  流速、圧力のデータの出力
C*****************************************************************       DO 330 I=0,M-1
         DO 320 J=0,N-1
C
           WRITE(9,*) U(I,J),V(I,J),P(I,J)
C
  320    CONTINUE
  330  CONTINUE
C
       CLOSE(9)
C
       RETURN
       END
C
C
C*********************************************************************
C*     局所平衡分布関数　　　　　　　　　　　　                      *
C*********************************************************************
C---------------------------------------------------------------------
C      ○局所平衡分布関数
C          初期状態での粒子分布および固体壁での粒子分布(境界条件)を
C          求めるために使用
C---------------------------------------------------------------------
       REAL FUNCTION EQUIV(U,V,U2,PI,LF0,C,ROU,K)
C---------------------------------------------------------------------
       REAL U,V,U2,PI,LF0,C,ROU
C
       IF(K.EQ.0) THEN
         EQUIV=0.5E0*ROU-ROU*U2/(C*C)
       ELSE
         CONST=C*(COS(PI*(K-1)/3.0E0)*U+SIN(PI*(K-1)/3.0E0)*V)
         EQUIV=ROU/(LF0+6.0E0)
     &        +ROU*CONST/(3.0E0*C*C)
     &        +2.0E0*ROU*(CONST*CONST-0.5E0*U2)/(3.0E0*C*C*C*C)
     &        +ROU*U2/(6.0E0*C*C)
       ENDIF
C
       RETURN
       END
