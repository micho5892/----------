C*********************************************************************
C*                                                                   *
C*         ２次元正方キャビティー内の自然対流 　　　　               *
C*               格子ボルツマン法（格子ＢＧＫモデル）                *
C*                       SHI-DE  FENG                                *
C*                                                                   *
C*                        THE GRADUATE SCHOOL OF KOBE UNIVERSITY     *
C*********************************************************************
C*********************************************************************
C　     Fr　――　赤粒子の分布関数
C　     Fb　――　青粒子の分布関数
C　     Rur　――　赤流体の密度　（入力はRur ＝Rub ＝Ru0）
C       Rub  ――  青粒子の密度
C　     U， V　――　流体の速度
C　     g　――　重力加速度
C　     X,Y　――　座標
C　     Rrj　――　並進後の赤粒子の密度
C　     Rbj　――　並進後の青粒子の密度
C　     Uj，Vj　――　並進後の流体の速度
C　     Flj　――　流れ関数
C　     Te　――　温度
C　　   入力file ―― lbconv.dat
C
C　     TT　――　計算の総時間STEPS
C　     TI　――　途中の出力STEPS
C       ｃ――　粒子の速度（無次元の速度ｃ＝１）
C　     Tk　――　緩和時間係数
C　     qr　――　熱源の熱伝達係数
C　     qb　――　冷たい壁の熱伝達係数
C       D ――　次元数（D=2）
C       b　――　運動粒子の方向数（ｂ＝６）
C       fu　―― 粘性係数
C　     PI  ――  円周率
C　     Fc　――　分布関数の係数
C　     Fd　――　分布関数の係数（運動の粒子に対応）
C　     Fo　――　分布関数の係数（静止の粒子に対応）
C
C　　　     
C　　　　N ＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿
C　　　　 |                                  |
C　　　　 |                                  |
C　　　　 |                                  |
C　　　　 |                                  |
C　　　　 |                                  |
C　　　　 |                                  |
C　　　L　|                                  |
C　　　　 |                                  |
C　　　　 |                                  |
C　　　　 |                                  |
C　　　　 |                                  |
C　　　　 |                                  |
C　　　　 |                                  |
C　　　　 |                                  |
C　　　　 |          M1         M2           |
C　　　　 |＿＿_＿＿＿|＿＿＿＿＿|＿＿_＿＿＿|                                  |
C        0             Heat source           M
C                            L 
C                    
C         M，N は偶数
C
C　　  出力files
C         ConvV　――　速度矢印分布
C　       ConvL　――　流線分布
C 　      ConvVy　――　センター上での速度Vy分布
C　       ConvT　――　温度分布
C　       ConvRB　――　 密度分布
C　       ConvR　――　 赤粒子の密度
C　       ConvB　――　 青粒子の密度
C　       ConvRL　――　等密度線分布
C
C*************************************************************
C
C
       COMMON /BK1/ Fr(0:200,0:200,0:6),Rur(0:200,0:200),II
       COMMON /BK2/ Fb(0:200,0:200,0:6),Rub(0:200,0:200)
       COMMON /BK3/ U(0:200,0:200),V(0:200,0:200),PI,Ru0
       COMMON /BK4/ M,N,M1,M2,TT,TI,Tk,Fc,Fd,Fo,c,qr,qb,g
       COMMON /BK5/ FLj(0:400,0:200),Rbj(0:400,0:200)
       COMMON /BK6/ X(0:400,0:200),Y(0:400,0:200),Rrj(0:400,0:200)
       COMMON /BK7/ Uj(0:400,0:200),Vj(0:400,0:200),Te(0:400,0:200)
C
       REAL  Fr,Fb,Rur,Rub,Rrj,Rbj,U,V,Uj,Vj,Flj,
     &       PI,Ru0,Tk,Fc,Fd,Fo,c,qr,qb,g,X,Y,Te
C
       INTEGER TT,TI
C
C
       PI=ATAN(1.0E0)*4.0E0
C
       CALL INPUT
C
        CALL EQUILIBRIUM
C
        DO 10 II=1,TT
C
         WRITE(*,*) 'TIME STEP = ',II
C
         IF (Fr(8,8,1).GE.1.0) GOTO 10000
C
         CALL COLLISION
C
         CALL INTERACTION
C
         CALL  CACUL1
C
         CALL CONVECTION(Fr,M,N)
         CALL CONVECTION(Fb,M,N)
C
         CALL  CACUL2
C
         CALL BOUNDARY
C
         IF((II.GE.45000).AND.(MOD(II,TI).EQ.0)) THEN
C
          CALL COMPUTING
C
         CALL OUTPUT
C
         ENDIF
C
10      CONTINUE
10000   CONTINUE
C
         STOP
         END
C
C
C
C********************************************************************
       SUBROUTINE INPUT
C********************************************************************
C
C
       COMMON /BK3/ U(0:200,0:200),V(0:200,0:200),PI,Ru0
       COMMON /BK4/ M,N,M1,M2,TT,TI,Tk,Fc,Fd,Fo,c,qr,qb,g
C
       INTEGER TT, TI
       REAL  Tk,Fc,Fd,Fo,D,b,c,qr,qb,g,Gr
C
        OPEN(8,FILE='lbconv.dat',STATUS='OLD')
         READ(8,*) M, N
         READ(8,*) M1,M2
         READ(8,*) D,b,c,Ru0
         READ(8,*) TT,TI,Tk
         READ(8,*) qr,qb,Gr
        CLOSE(8)
C
C
           Ls=real(M)
           Fu=(Tk-0.50e0)/4.0e0
           g=Gr*Fu*Fu/(2.0e0*COS(PI/6.0e0)*Ls*Ls*Ls)
C
             Fc=0.50e0*(D+2.0e0)/(c*c)
             Fd=D/(b*(D+2.0e0))
             Fo=2.0e0/(D+2.0e0)
C
       RETURN
       END
C
C
C********************************************************************
       SUBROUTINE EQUILIBRIUM
C********************************************************************
C
       COMMON /BK1/ Fr(0:200,0:200,0:6),Rur(0:200,0:200),II
       COMMON /BK2/ Fb(0:200,0:200,0:6),Rub(0:200,0:200)
       COMMON /BK3/ U(0:200,0:200),V(0:200,0:200),PI,Ru0
       COMMON /BK4/ M,N,M1,M2,TT,TI,Tk,Fc,Fd,Fo,c,qr,qb,g
C
       REAL Tk,Fr,Fb,U,V,Ui,Vi,V2i,Rur,Rub,Ru0,PI,Fc,Fd,Fo
C
       Ui=0.0e0
       Vi=0.0e0
       V2i=Ui*Ui+Vi*Vi
C
       DO 80 J=0,N
       DO 70 I=0, M
C
         U(I,J)=0.0
         V(I,J)=0.0
         Rur(I,J)=Ru0
         Rub(I,J)=Ru0
C
       DO 60 K=0,6
C
          Fr(I,J,K)=Feqi(Fc,Fd,Fo,PI,Rur(I,J),Ui,Vi,V2i,K)
          Fb(I,J,K)=Feqi(Fc,Fd,Fo,PI,Rub(I,J),Ui,Vi,V2i,K)
C
  60    CONTINUE
  70    CONTINUE
  80    CONTINUE
C
        WRITE(*,*) 'Tk = ',Tk, 'Gr = ',Gr, 'g = ',g
C
        RETURN
        END
C
C
C*********************************************************************
       SUBROUTINE COLLISION
C*********************************************************************
C
C
       COMMON /BK1/ Fr(0:200,0:200,0:6),Rur(0:200,0:200),II
       COMMON /BK2/ Fb(0:200,0:200,0:6),Rub(0:200,0:200)
       COMMON /BK3/ U(0:200,0:200),V(0:200,0:200),PI,Ru0
       COMMON /BK4/ M,N,M1,M2,TT,TI,Tk,Fc,Fd,Fo,c,qr,qb,g
C
       REAL  Tk,Fr,Fb,Fkri,Fkbi,U,V,V2i,Rur,Rub,PI,Fc,Fd,Fo
C
       DO 110 J=1,N-1
       DO 100 I=0,M-1
C
           IF ((I.EQ.0).AND.(MOD(J,2).EQ.0)) GOTO 100
C
           V2i=U(I,J)*U(I,J)+V(I,J)*V(I,J)
           DO 98 K=0,6
C
             Fkri=Feqi(Fc,Fd,Fo,PI,Rur(I,J),U(I,J),V(I,J),V2i,K)
             Fkbi=Feqi(Fc,Fd,Fo,PI,Rub(I,J),U(I,J),V(I,J),V2i,K)
             Fr(I,J,K)=Fr(I,J,K)-(Fr(I,J,K)-Fkri)/Tk
             Fb(I,J,K)=Fb(I,J,K)-(Fb(I,J,K)-Fkbi)/Tk
C
  98    CONTINUE
 100    CONTINUE
 110    CONTINUE
C
        RETURN
        END
C
C
C**********************************************************************
         SUBROUTINE  INTERACTION
C**********************************************************************
C
C
       COMMON /BK1/ Fr(0:200,0:200,0:6),Rur(0:200,0:200),II
       COMMON /BK2/ Fb(0:200,0:200,0:6),Rub(0:200,0:200)
       COMMON /BK3/ U(0:200,0:200),V(0:200,0:200),PI,Ru0
       COMMON /BK4/ M,N,M1,M2,TT,TI,Tk,Fc,Fd,Fo,c,qr,qb,g
C
       REAL Fr,Fb,Fbi,Fbj,qb,qr,g
C
C
       DO 892  J=1, N-1
       DO 891  I=0, M-1
C
           IF ((I.EQ.0).AND.(MOD(J,2).EQ.0)) GOTO 891
C
C
           Fbi=Fb(I,J,2)*g
           Fbj=Fb(I,J,3)*g
C
           Fb(I,J,2)=Fb(I,J,2)-Fbi
           Fb(I,J,6)=Fb(I,J,6)+Fbi
C
           Fb(I,J,3)=Fb(I,J,3)-Fbj
           Fb(I,J,5)=Fb(I,J,5)+Fbj
C
 891   CONTINUE
 892   CONTINUE
C
C
C
       RETURN
       END
C
C
C
C*********************************************************************
       SUBROUTINE BOUNDARY
C*********************************************************************
C
C
       COMMON /BK1/ Fr(0:200,0:200,0:6),Rur(0:200,0:200),II
       COMMON /BK2/ Fb(0:200,0:200,0:6),Rub(0:200,0:200)
       COMMON /BK3/ U(0:200,0:200),V(0:200,0:200),PI,Ru0
       COMMON /BK4/ M,N,M1,M2,TT,TI,Tk,Fc,Fd,Fo,c,qr,qb,g
C
       REAL  Fr,Fb,qr,qb
C
        DO 7 I=1,M-1
C
         Fr(I,N,6)=Fr(I,N,3)*(1.0E0-qb)
         Fr(I,N,5)=Fr(I,N,2)*(1.0E0-qb)
         Fb(I,N,6)=Fb(I,N,3)+Fr(I,N,3)*qb
         Fb(I,N,5)=Fb(I,N,2)+Fr(I,N,2)*qb
C
 7      CONTINUE
C
C----------------------------------------------------[BOTTOM BOUNDARY]
C
       DO 8 I=1, M-1
C
         IF ((I.GE.M1).AND.(I.LE.M2))   THEN
C
           Fr(I,0,2)=Fr(I,0,6)+Fb(I,0,6)*qr
           Fr(I,0,3)=Fr(I,0,5)+Fb(I,0,5)*qr
           Fb(I,0,2)=Fb(I,0,6)*(1.0E0-qr)
           Fb(I,0,3)=Fb(I,0,5)*(1.0E0-qr)
C
         ELSE
           Fr(I,0,2)=Fr(I,0,5)*(1.0-qb)
           Fr(I,0,3)=Fr(I,0,6)*(1.0-qb)
           Fb(I,0,2)=Fb(I,0,5)+Fr(I,0,5)*qb
           Fb(I,0,3)=Fb(I,0,6)+Fr(I,0,6)*qb
         END IF
C
 8     CONTINUE
C
C------------------------------------------------[LEFT&RIGHT BOUNDARY]
C
       DO 9 J=2,N-2,2
C
         Fr(0,J,1)=Fr(0,J,4)*(1.0-qb)
         Fb(0,J,1)=Fb(0,J,4)+Fr(0,J,4)*qb
         Fr(0,J,2)=Fr(0,J,5)*(1.0-qb)
         Fb(0,J,2)=Fb(0,J,5)+Fr(0,J,5)*qb
         Fr(0,J,6)=Fr(0,J,3)*(1.0-qb)
         Fb(0,J,6)=Fb(0,J,3)+Fr(0,J,3)*qb
C
         Fr(M,J,4)=Fr(M,J,1)*(1.0-qb)
         Fb(M,J,4)=Fb(M,J,1)+Fr(M,J,1)*qb
         Fr(M,J,3)=Fr(M,J,6)*(1.0-qb)
         Fb(M,J,3)=Fb(M,J,6)+Fr(M,J,6)*qb
         Fr(M,J,5)=Fr(M,J,2)*(1.0-qb)
         Fb(M,J,5)=Fb(M,J,2)+Fr(M,J,2)*qb
C
 9     CONTINUE
C
C---------------------------FOUR CORNER
C
          Fr(0,N,1)=Fr(0,N,4)*(1.0E0-qb)
          Fr(0,N,6)=Fr(0,N,3)*(1.0E0-qb)
          Fb(0,N,1)=Fb(0,N,4)+Fr(0,N,4)*qb
          Fb(0,N,6)=Fb(0,N,3)+Fr(0,N,3)*qb
C
          Fr(M,N,4)=Fr(M,N,1)*(1.0E0-qb)
          Fr(M,N,5)=Fr(M,N,2)*(1.0E0-qb)
          Fb(M,N,4)=Fb(M,N,1)+Fr(M,N,1)*qb
          Fb(M,N,5)=Fb(M,N,2)+Fr(M,N,2)*qb
C
          Fr(0,0,1)=Fr(0,0,4)*(1.0E0-qb)
          Fr(0,0,2)=Fr(0,0,5)*(1.0E0-qb)
          Fb(0,0,1)=Fb(0,0,4)+Fr(0,0,4)*qb
          Fb(0,0,2)=Fb(0,0,5)+Fr(0,0,5)*qb
C
          Fr(M,0,4)=Fr(M,0,1)*(1.0E0-qb)
          Fr(M,0,3)=Fr(M,0,6)*(1.0E0-qb)
          Fb(M,0,4)=Fb(M,0,1)+Fr(M,0,1)*qb
          Fb(M,0,3)=Fb(M,0,6)+Fr(M,0,6)*qb
C
C
C
       RETURN
       END
C
C**********************************************************************
         SUBROUTINE  CACUL1
C**********************************************************************
C
       COMMON /BK1/ Fr(0:200,0:200,0:6),Rur(0:200,0:200),II
       COMMON /BK2/ Fb(0:200,0:200,0:6),Rub(0:200,0:200)
       COMMON /BK3/ U(0:200,0:200),V(0:200,0:200),PI,Ru0
       COMMON /BK4/ M,N,M1,M2,TT,TI,Tk,Fc,Fd,Fo,c,qr,qb,g
C
       REAL  Fr,Fb,U,V,Ui,Vi,Rur,Rub,PI,Rr,Rb
C
C
       DO 9  J=1, N-1
       DO 8  I=0, M-1
C
           IF ((I.EQ.0).AND.(MOD(J,2).EQ.0)) GOTO  8
C
C
            Ui=0.0E0
            Vi=0.0E0
            Rr=Fr(I,J,0)
            Rb=Fb(I,J,0)
C
           DO 7 K=1,6
C
             Rr=Rr+Fr(I,J,K)
             Rb=Rb+Fb(I,J,K)
             Ui=Ui+(Fr(I,J,K)+Fb(I,J,K))*c*COS(PI*(K-1)/3.0E0)
             Vi=Vi+(Fr(I,J,K)+Fb(I,J,K))*c*SIN(PI*(K-1)/3.0E0)
C
 7      CONTINUE
C
           U(I,J)=Ui/(Rr+Rb)
           V(I,J)=Vi/(Rr+Rb)
           Rur(I,J)=Rr
           Rub(I,J)=Rb
C
 8      CONTINUE
 9      CONTINUE
C
        RETURN
        END
C
C
C**********************************************************************
         SUBROUTINE  CACUL2
C**********************************************************************
C
       COMMON /BK1/ Fr(0:200,0:200,0:6),Rur(0:200,0:200),II
       COMMON /BK2/ Fb(0:200,0:200,0:6),Rub(0:200,0:200)
       COMMON /BK3/ U(0:200,0:200),V(0:200,0:200),PI,Ru0
       COMMON /BK4/ M,N,M1,M2,TT,TI,Tk,Fc,Fd,Fo,c,qr,qb,g
C
       REAL Fr,Fb,U,V,Ui,Vi,Rur,Rub,PI,Rr,Rb
C
C
       DO 9  J=1, N-1
       DO 8  I=0, M-1
C
         IF ((I.EQ.0).AND.(MOD(J,2).EQ.0)) GOTO 8
C
C
            Ui=0.0E0
            Vi=0.0E0
            Rr=Fr(I,J,0)
            Rb=Fb(I,J,0)
C
           DO 7 K=1,6
C
             Rr=Rr+Fr(I,J,K)
             Rb=Rb+Fb(I,J,K)
             Ui=Ui+(Fr(I,J,K)+Fb(I,J,K))*c*COS(PI*(K-1)/3.0E0)
             Vi=Vi+(Fr(I,J,K)+Fb(I,J,K))*c*SIN(PI*(K-1)/3.0E0)
C
 7      CONTINUE
C
           U(I,J)=0.50e0*(U(I,J)+Ui/(Rr+Rb))
           V(I,J)=0.50e0*(V(I,J)+Vi/(Rr+Rb))
           Rur(I,J)=0.50e0*(Rur(I,J)+Rr)
           Rub(I,J)=0.50e0*(Rub(I,J)+Rb)
C
 8     CONTINUE
 9     CONTINUE
C
        RETURN
        END
C
C
C*********************************************************************
         SUBROUTINE  CONVECTION(F,M,N)
C*********************************************************************
C
C
         REAL F(0:200,0:200,0:6)
C
C
C------------------1-DIRECTION
C
         DO 7 J=N, 0, -2
         DO 6 I=M, 1, -1
C
             F(I,J,1)=F(I-1,J,1)
C
 6      CONTINUE
 7      CONTINUE
C
         DO 9 J=N-1, 1, -2
         DO 8 I=M-1, 1, -1
C
             F(I,J,1)=F(I-1,J,1)
C
 8     CONTINUE
 9     CONTINUE
C
C------------------2-DIRECTION
C
         DO 17 J=N, 2, -2
         DO 16 I=M, 1, -1
C
           F(I,J,2)=F(I-1,J-1,2)
           F(I-1,J-1,2)=F(I-1,J-2,2)
C
 16      CONTINUE
 17      CONTINUE
C
C
C-------------------3-DIRECTION
C
         DO 19 J=N, 2, -2
         DO 18 I=0, M-1
C
            F(I,J,3)=F(I,J-1,3)
            F(I,J-1,3)=F(I+1,J-2,3)
C
 18     CONTINUE
 19     CONTINUE
C
C
C------------------4-DIRECTION
C
       DO 27 J=N ,0 ,-2
       DO 26 I=0 ,M-1
C
             F(I,J,4)=F(I+1,J,4)
C
 26     CONTINUE
 27     CONTINUE
C
       DO 29 J=N-1 ,1 ,-2
       DO 28 I=0 ,M-2
C
             F(I,J,4)=F(I+1,J,4)
C
 28      CONTINUE
 29      CONTINUE
C
C------------------5-DIRECTION
C
         DO 37 J=0, N-2, 2
         DO 36 I=0, M-1
C
            F(I,J,5)=F(I,J+1,5)
            F(I,J+1,5)=F(I+1,J+2,5)
C
 36     CONTINUE
 37     CONTINUE
C
C------------------6-DIRECTION
C
         DO 39  J=0, N-2, 2
         DO 38  I=M, 1,  -1
C
           F(I,J,6)=F(I-1,J+1,6)
           F(I-1,J+1,6)=F(I-1,J+2,6)
C
 38      CONTINUE
 39      CONTINUE
C
          RETURN
          END
C
C
C
C**********************************************************************
         REAL  FUNCTION  Feqi(Fc,Fd,Fo,PI,Ru,U,V,V2,K)
C**********************************************************************
C
         REAL  Fc,Fd,Fo,PI,Ru,U,V,V2
C
C
            FcV2=Fc*V2
          IF(K.EQ.0) THEN
            Feqi=Fo*Ru*(1.0e0-FcV2)
C
          ELSE
            VRi=U*COS(PI*(K-1)/3.0E0)+V*SIN(PI*(K-1)/3.0E0)
            FcVRi=Fc*VRi
C
            Feqi=Fd*Ru*(1.0e0-FcV2+2.0e0*FcVRi*(1.0e0+FcVRi
     $          -FcV2+2.0e0*FcVRi*FcVRi/3.0e0))
C
          END IF
C
         RETURN
         END
C
C
C
C*********************************************************************
       SUBROUTINE COMPUTING
C*********************************************************************
C
       COMMON /BK1/ Fr(0:200,0:200,0:6),Rur(0:200,0:200),II
       COMMON /BK2/ Fb(0:200,0:200,0:6),Rub(0:200,0:200)
       COMMON /BK3/ U(0:200,0:200),V(0:200,0:200),PI,Ru0
       COMMON /BK4/ M,N,M1,M2,TT,TI,Tk,Fc,Fd,Fo,c,qr,qb,g
       COMMON /BK5/ FLj(0:400,0:200),Rbj(0:400,0:200)
       COMMON /BK6/ X(0:400,0:200),Y(0:400,0:200),Rrj(0:400,0:200)
       COMMON /BK7/ Uj(0:400,0:200),Vj(0:400,0:200),Te(0:400,0:200)
C
       REAL  Fr,Fb,Rur,Rub,Rrj,Rbj,U,V,Uj,Vj,Flj,PI,X,Y,Te
C
C
        DO 9 J=0, N
        DO 8 I=0, 2*M
C
            X(I,J)=real(I)/2.0e0
            Y(I,J)=J*SIN(PI/3.0E0)
C
 8      CONTINUE
 9      CONTINUE
C
C
        DO 17 J=0, N, 2
        DO 16 I=0, M
C
                L=2*I
                Uj(L,J)=U(I,J)
                Vj(L,J)=V(I,J)
                Rrj(L,J)=Rur(I,J)
                Rbj(L,J)=Rub(I,J)
C
 16     CONTINUE
 17     CONTINUE
C
         DO 19 J=1, N-1, 2
         DO 18 I=0, M-1
C
               L=2*I+1
               Uj(L,J)=U(I,J)
               Vj(L,J)=V(I,J)
               Rrj(L,J)=Rur(I,J)
               Rbj(L,J)=Rub(I,J)
C
 18     CONTINUE
 19     CONTINUE
C
         DO 26 J=0, N
C
                Uj(0,J)=0.0e0
                Vj(0,J)=0.0e0
                Uj(2*M,J)=0.0e0
                Vj(2*M,J)=0.0e0
C
26      CONTINUE
C
         DO 27 I=1, 2*M-1,2
C
C
                Uj(I,0)=0.0e0
                Vj(I,0)=0.0e0
                Uj(I,N)=0.0e0
                Vj(I,N)=0.0e0
                Rrj(I,0)=(Rrj(I-1,0)+Rrj(I+1,0))/2.0e0
                Rbj(I,0)=(Rbj(I-1,0)+Rbj(I+1,0))/2.0e0
                Rrj(I,N)=(Rrj(I-1,N)+Rrj(I+1,N))/2.0e0
                Rbj(I,N)=(Rbj(I-1,N)+Rbj(I+1,N))/2.0e0
C
 27     CONTINUE
C
         DO 29 J=2, N-2, 2
         DO 28 L=1, 2*M-1, 2
C
                Uj(L,J)=0.50e0*(Uj(L-1,J)+Uj(L+1,J))
                Vj(L,J)=0.50e0*(Vj(L-1,J)+Vj(L+1,J))
                Rrj(L,J)=(Rrj(L-1,J)+Rrj(L+1,J))/2.0e0
                Rbj(L,J)=(Rbj(L-1,J)+Rbj(L+1,J))/2.0e0
C
 28     CONTINUE
 29     CONTINUE
C
         DO 37 J=1, N-1, 2
         DO 36 L=2, 2*M-2, 2
C
                Uj(L,J)=0.50e0*(Uj(L-1,J)+Uj(L+1,J))
                Vj(L,J)=0.50e0*(Vj(L-1,J)+Vj(L+1,J))
                Rrj(L,J)=(Rrj(L-1,J)+Rrj(L+1,J))/2.0e0
                Rbj(L,J)=(Rbj(L-1,J)+Rbj(L+1,J))/2.0e0
C
 36     CONTINUE
 37     CONTINUE
C
          DO 38 J=1, N-1, 2
C
           Rrj(0,J)=(Rrj(0,J-1)+Rrj(0,J+1))/2.0e0
           Rbj(0,J)=(Rbj(0,J-1)+Rbj(0,J+1))/2.0e0
C
           Rrj(2*M,J)=(Rrj(2*M,J-1)+Rrj(2*M,J+1))/2.0e0
           Rbj(2*M,J)=(Rbj(2*M,J-1)+Rbj(2*M,J+1))/2.0e0
C
 38      CONTINUE
C
         DO 39 J=2, N-2, 2
C
           Uj(1,J)=(Uj(1,J-1)+Uj(1,J+1))/2.0e0
           Vj(1,J)=(Vj(1,J-1)+Vj(1,J+1))/2.0e0
           Rrj(1,J)=(Rrj(1,J-1)+Rrj(1,J+1))/2.0e0
           Rbj(1,J)=(Rbj(1,J-1)+Rbj(1,J+1))/2.0e0
C
           Uj(2*M-1,J)=(Uj(2*M-1,J-1)+Uj(2*M-1,J+1))/2.0e0
           Vj(2*M-1,J)=(Vj(2*M-1,J-1)+Vj(2*M-1,J+1))/2.0e0
           Rrj(2*M-1,J)=(Rrj(2*M-1,J-1)+Rrj(2*M-1,J+1))/2.0e0
           Rbj(2*M-1,J)=(Rbj(2*M-1,J-1)+Rbj(2*M-1,J+1))/2.0e0
C
 39     CONTINUE
C
         DO 46 L=0, 2*M
C
            FLj(L,0)=0.0E0
C
 46     CONTINUE
C
         DO 47 J=1, N
C
            FLj(0,J)=FLj(0,J-1)+Uj(0,J)*SIN(PI/3.0E0)
C
 47     CONTINUE
C
         DO 49 L=1, 2*M
         DO 48 J=1, N
C
           FLj(L,J)=FLj(L,J-1)+Uj(L,J)*SIN(PI/3.0E0)
C
 48     CONTINUE
 49     CONTINUE
C
         DO 59 I=1, 2*M-1
         DO 58 J=1, N-1
C
           Te(I,J)=Rrj(I,J)/(Rrj(M,0)+Rbj(M,0))
C
 58     CONTINUE
 59     CONTINUE
C
         RETURN
         END
C
C
C
C*********************************************************************
       SUBROUTINE OUTPUT
C*********************************************************************
C
C
       COMMON /BK1/ Fr(0:200,0:200,0:6),Rur(0:200,0:200),II
       COMMON /BK2/ Fb(0:200,0:200,0:6),Rub(0:200,0:200)
       COMMON /BK3/ U(0:200,0:200),V(0:200,0:200),PI,Ru0
       COMMON /BK4/ M,N,M1,M2,TT,TI,Tk,Fc,Fd,Fo,c,qr,qb,g
       COMMON /BK5/ FLj(0:400,0:200),Rbj(0:400,0:200)
       COMMON /BK6/ X(0:400,0:200),Y(0:400,0:200),Rrj(0:400,0:200)
       COMMON /BK7/ Uj(0:400,0:200),Vj(0:400,0:200),Te(0:400,0:200)
       CHARACTER   NAME*20
C
       REAL  Rrj,Rbj,Uj,Vj,Flj,X,Y,Te
C
C
C
         Nc=N/2
C
        TIME=FLOAT(II)+0.001
        KK=INT(TIME)
        K10=KK/1000000
        KK9=KK-K10*1000000
        KK8=KK9/100000
        KK7=KK9-KK8*100000
        KK6=KK7/10000
        KK5=KK7-KK6*10000
        KK4=KK5/1000
c        KK3=KK5-KK4*1000
c        KK2=KK3/100
C
        NAME='ConvV'//CHAR(KK8+48)//CHAR(KK6+48)//CHAR(KK4+48)//'.dat'
C
        OPEN(9,FILE=NAME,STATUS='UNKNOWN')
C
         DO 29 J=1,N-1
         DO 28 I=1,  2*M-1
C
         WRITE(9,*)  X(I,J),Y(I,J), Uj(I,J),Vj(I,J)
C
 28    CONTINUE
 29    CONTINUE
        CLOSE(9)
C
        NAME='ConvL'//CHAR(KK8+48)//CHAR(KK6+48)//CHAR(KK4+48)//'.dat'
C
        OPEN(9,FILE=NAME,STATUS='UNKNOWN')
         WRITE(9,*)  2*M-1, N
C
         DO 39 J=1,N
         DO 38 I=1,  2*M-1
C
         WRITE(9,*)  X(I,J-1), Y(I,J-1), FLj(I,J-1)
C
 38    CONTINUE
 39    CONTINUE
        CLOSE(9)
C
        NAME='ConvVy'//CHAR(KK8+48)//CHAR(KK6+48)//CHAR(KK4+48)//'.dat'
        OPEN(9,FILE=NAME,STATUS='UNKNOWN')
C
         DO 48 I=0, 2*M
C
         WRITE(9,*)    X(I,Nc)/X(2*M,Nc), Vj(I,Nc)/Vj(M,Nc)
48      CONTINUE
        CLOSE(9)
C
         NAME='ConvT'//CHAR(KK8+48)//CHAR(KK6+48)//CHAR(KK4+48)//'.dat'
         OPEN(9,FILE=NAME,STATUS='UNKNOWN')
C
         WRITE(9,*)  2*M-1, N-1
C
         DO 59 J=1,N-1
         DO 58 I=1,  2*M-1
C
           WRITE(9,*)  X(I,J), Y(I,J), Te(I,J)
C
 58    CONTINUE
 59    CONTINUE
         CLOSE(9)
C
        NAME='ConvRB'//CHAR(KK8+48)//CHAR(KK6+48)//CHAR(KK4+48)//'.dat'
         OPEN(9,FILE=NAME,STATUS='UNKNOWN')
C
C
         DO 78 J=1,  N
C
           Rm=Rrj(M,J)+Rbj(M,J)
C
           WRITE(9,*)  Rm, Y(M,J)/Y(M,N)
C
78    CONTINUE
         CLOSE(9)
C
         NAME='ConvR'//CHAR(KK8+48)//CHAR(KK6+48)//CHAR(KK4+48)//'.dat'
         OPEN(9,FILE=NAME,STATUS='UNKNOWN')
C
C
         DO 88 J=1,  N
C
           WRITE(9,*) Rrj(M,J), Y(M,J)/Y(M,N)
C
88    CONTINUE
         CLOSE(9)
C
        NAME='ConvB'//CHAR(KK8+48)//CHAR(KK6+48)//CHAR(KK4+48)//'.dat'
         OPEN(9,FILE=NAME,STATUS='UNKNOWN')
C
C
         DO 97 J=1,  N
C
           WRITE(9,*)  Rbj(M,J), Y(M,J)/Y(M,N)
C
97     CONTINUE
         CLOSE(9)
C
        NAME='ConvRL'//CHAR(KK8+48)//CHAR(KK6+48)//CHAR(KK4+48)//'.dat'
         OPEN(9,FILE=NAME,STATUS='UNKNOWN')
C
         WRITE(9,*)  2*M-1, N-1
C
          DO 99 J=1,  N-1
          DO 98 I=1,  2*M-1
C
            RNc=Rrj(I,J)+Rbj(I,J)
           WRITE(9,*)  X(I,J)/X(2*M,N), Y(I,J)/Y(M,N), RNc
C
98      CONTINUE
99      CONTINUE
         CLOSE(9)
C
C
         RETURN
         END
C
C
