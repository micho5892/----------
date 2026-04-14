
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+                                                                     +
+       格子気体法による矩形容器内で回転運動する平板周りの流れ　　    +
+                                           (ブール演算)              +
+                                PROGRAMED BY H.MURATA  (KOBE UNIV.)  +
+                                                      (1993)         +
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

/************************************************************************
  セルオートマトンプログラムの入力パラメータの説明


プログラムにおいて、入力パラメータを下記とした場合のタイムステップ
の内訳を図１に示す。

１．データを取る間隔内の時間ステップ数（○：lll = 5　）
２．疎視化の時間ステップ数（●：KK = 3）
３．全時間ステップ数(データを取る数（hh = 5）
４．最初に平衡状態を得るための時間ステップ数（△：fermi = 9）
５．データファイルの名前(ss = File）　
　（出力ファイルのヘッダー名：File-***.dat)

（１） 流れ場を平衡状態にさせるため、平板を移動させずに任意のタイム
ステップ（"fermi"：流れ場が十分落ち着く回数）
粒子の衝突・並進を行う。
（２）（１）終了後、平板の移動を開始する。
（３） 流れ場を見る（流速等を出力する）間隔は（lll＋kk）回毎であるが、
そのうちのlll回は単にタイムを進めるだけであり、疎視化して流速等を計算
するのに使用する時間平均タイムステップがkkである。
（４）（３）の操作をhh回繰り返して計算が終了する。全部のタイムステップ
は fermi＋（lll＋kk）×hh 


　平衡状態にする　　　　　　　　　　　　　　　　　平板周りの流れを観測する
　タイムステップ　　　　　　　　　　　　　　　　　　　タイムステップ
　　　　　　　　　
　　　　　　平板の移動開始
　　　　　　　　　↓
△△△△△△△△△○○○○○●●●○○○○○●●●○○○○○●●●○○○○○●●●○○○○○●●●
　　　　　　　　　　　　　　　　↓　　　　　　　↓　　　　　　　↓　　　　　　　↓　　　　　　　↓　　　　　　　
　　　　　　　　　　　　　 　速度出力　　　  速度出力        速度出力        速度出力        速度出力
　　　　　　　　　　　　　  （疎視化）      （疎視化）      （疎視化）      （疎視化）      （疎視化）


　　　　　　　　　　　　　　　　　　図１　タイムステップの内訳

*************************************************************************/

#include <stdio.h>
#include <string.h>           /***   UNIX の場合 <strings.h> とする  ***/
#include <math.h>

#define PC_CHECK 0
#define CHECK0 0
#define CHECK1 0
#define M 2560  /*** 計算格子数　横軸 ***/
#define N 1920  /*** 計算格子数　縦軸 ***/
#define P 64    /*** 疎視化格子数　横軸 ***/
#define Q 64    /*** 疎視化格子数　縦軸 ***/
#define R 1
#define S 32
#define L 64
#define SS 16
#define J 4      /* L/SS */
#define OMEGA 1.0/500.0   /*** 平板の回転角速度 ***/
#define CELL 350
#define LL 1664501
#define LAMBDA 1229
#define MU 351750
#define PI 3.14159265   /***  円周率  ***/
#define DUMMY 0.00000001

void collision();
void advance_of_particles();
void input();
void initialization();
void initial_distribution();
void fermi_dirac();
void fermi_dirac_op();
void initialize_average();
void average();
void write_on_file();
void fhp_1();
void inner_advance();
void wing_coll_before();
void velocity_of_the_flow();
void wing_coll_after();
int urand();
void itoan();
void check();
void quick_sort();
void cal0_1();
void cal1_2();
void cal2_3();
void cal3_4();
void cal4_5();
void cal5_0();
void t_bottom_coll();
void t_top_coll();
void t_right_coll();
void t_left_coll();
void corner_coll();

struct bit {unsigned a32:1; unsigned a31:1; unsigned a30:1;
            unsigned a29:1; unsigned a28:1; unsigned a27:1;
            unsigned a26:1; unsigned a25:1; unsigned a24:1;
            unsigned a23:1; unsigned a22:1; unsigned a21:1;
            unsigned a20:1; unsigned a19:1; unsigned a18:1;
            unsigned a17:1; unsigned a16:1; unsigned a15:1;
            unsigned a14:1; unsigned a13:1; unsigned a12:1;
            unsigned a11:1; unsigned a10:1; unsigned a9:1;
            unsigned a8:1; unsigned a7:1; unsigned a6:1;
            unsigned a5:1; unsigned a4:1; unsigned a3:1;
            unsigned a2:1; unsigned a1:1;};
struct bit1 {unsigned f:1; unsigned e:1; unsigned d:1; unsigned c:1;
             unsigned b:1; unsigned a:1; unsigned g:1; unsigned h:1;};
struct bit2{unsigned f:1; unsigned e:1; unsigned d:1; unsigned c:1;
             unsigned b:1; unsigned a:1; unsigned g:1; unsigned h:1;};
union body {struct bit bool; unsigned long z;}a[M][N][6];
union body1 {struct bit1 bool1; unsigned char z1;}a1[N];
union body2 {struct bit2 bool2; unsigned char z2;}aa[CELL][J];

int kk;
int hh;
unsigned lll;
unsigned fermi;
char ss[20];

double at[P][Q];
double ax[P][Q];
double ay[P][Q];
unsigned long tl;
/*--------------    境界条件の設定   --------------*/
int pxt[CELL][J],pyt[CELL][J],sh[CELL][J],am[CELL][J];
long se[CELL];
double theta;
int collnum;
struct {int px; int py; double pd;}mura[CELL];
/*-------------------------------------------------------------------*/
void main()
{
        int l,h,k;
        unsigned long circ;

        input();
        initialization();
        initial_distribution();
        fermi_dirac();
        write_on_file(0);
        
        circ=1;
        for(h=1;h<=hh;h++){
            l=0;
            initialize_average();
            for(k=1;k<=kk;k++) {
                do{
                    tl=0;
                    l++; 
                    theta=OMEGA*(double)circ;
                    printf("l=%d\n",l);
                    collision();
#if CHECK0
                    check(1);
#endif
                    advance_of_particles();
#if CHECK0
                    check(2);
#endif
                    circ++;
                }while(l<=lll);
                average();
            }
            write_on_file(h);
        }
}
/**********************************************************************
                      サブルーチン FUNCTION
**********************************************************************/
void collision()
{
        if(theta<(PI/3.0)){
            collnum=0;
        }
        else if(theta<(2.0*PI/3.0)){
            collnum=1;
        }
        else if(theta<PI){
            collnum=2;
        }
        else{
            collnum=3;
        }
        switch(collnum)
        {
            case 0:
            case 1:
            case 2:    wing_coll_before();
                       fhp_1();
                       wing_coll_after();
                       break;
            case 3:    fhp_1();
                       break;
            default:   printf("angle step error!");
        }
        t_bottom_coll();
        t_top_coll();
        t_left_coll();
        t_right_coll();
        corner_coll();
}
void advance_of_particles()
{
        inner_advance();
}
/*******************************************************************
　　　データ入力　（かっこ内の数値が基本）
*******************************************************************/
void input()
{
        printf(" データを取る間隔内の時間ステップ数 (=5) ?");
        scanf("%d",&lll);
        printf(" 疎視化の時間ステップ数 (=3) ?");
        scanf("%d",&kk);
        printf("全時間ステップ数(データを取る数（= 5） ?");
        scanf("%d",&hh);
        printf("最初に平衡状態を得るための時間ステップ数 (=9) ?");
        scanf("%d",&fermi);
        printf(" データファイルの名前(ss = File 例えば rot）?");
        scanf("%s",ss);
        strcat(ss,"-");
}
void initialization()
{
        int x,y,i;

        for(x=0;x<=M-1;x++){
            for(y=0;y<=N-1;y++){
                for(i=0;i<=5;i++){
                    a[x][y][i].z=(long)0x0;
                }
            }
        }
        for(y=0;y<=N-1;y++){
            a1[y].z1=0x0;
        }
        for(i=0;i<CELL;i++){
            for(y=0;y<J;y++){
                aa[i][y].z2=0x0;
            }
        }
}
void initial_distribution()
{
        int ii;
        int ll;
        int x,y;
        int i,j,k;
        double num,den;
        unsigned long tl;
        unsigned long r[6];
        unsigned itot[P][Q][6];

        num=32*R*S;
        tl=0;
        
        for(x=0;x<=M-1;x++) {
            for(y=1;y<=N-2;y++) {
                for(k=0;k<=5;k++){
                    for(i=0;i<6;i++){
                        ll=urand(4);
                        switch(ll)
                        {
                            case 0:         r[i]=(long)0x1;
                                            break;
                            case 1:         r[i]=(long)0x2;
                                            break;
                            case 2:         r[i]=(long)0x4;
                                            break;
                            case 3:         r[i]=(long)0x8;
                        }
                    }
                    a[x][y][k].z=r[0]<<24|r[1]<<20|r[2]<<16|
                                 r[3]<<12|r[4]<<8|r[5]<<4|0x0;
                }
            }
        }
        for(i=0;i<P;i++) {
            for(j=0;j<Q;j++) {
                for(k=0;k<6;k++) {
                    itot[i][j][k]=0;
                }
                for(x=R*i;x<=R*(i+1)-1;x++) {
                    for(y=S*j;y<=S*(j+1)-1;y++) {
                        for(ii=0;ii<32;ii++){
                            for(k=0;k<6;k++){
                                itot[i][j][k]+=(a[x][y][k].z<<ii)>>31;
                            }
                        }
                    }
                }
                printf("density[%d][%d]=%lf\n",i,j,(itot[i][j][0]+
                        itot[i][j][1]+itot[i][j][2]+itot[i][j][3]+
                        itot[i][j][4]+itot[i][j][5])/num);
                tl+=itot[i][j][0]+itot[i][j][1]+itot[i][j][2]+
                    itot[i][j][3]+itot[i][j][4]+itot[i][j][5];
            }
        }
        den=(double)tl/(double)(M*32)/(double)N;
        printf("initial total number=%lu    \n",tl);
        printf("initial density=%lf\n",den);
}
void fermi_dirac()
{
        unsigned k;

        for(k=1;k<=fermi;k++){
            printf("%d\n",k);
            fermi_dirac_op();
        }
        initialize_average();
        for(k=fermi+1;k<=fermi+kk;k++){
            printf("%d\n",k);
            tl=0;
            fermi_dirac_op();
            average();
        }
        check(0);
}
void fermi_dirac_op()
{
            fhp_1();
            t_bottom_coll();
            t_top_coll();
            t_right_coll();
            t_left_coll();
            corner_coll();
            inner_advance();
}
void initialize_average()
{
        int i,j;
        
        for(i=0;i<P;i++) {
            for(j=0;j<Q;j++) {
                at[i][j]=0.0;
                ax[i][j]=0.0;
                ay[i][j]=0.0;
            }
        }
}
void average()
{
        int i,j,k;
        int x,y,ii;
        double num;
        double total[P][Q];
        double xi[P][Q],yi[P][Q];
        double itot[P][Q][6];

        num=32*R*S;
        for(i=0;i<P;i++){
            for(j=0;j<Q;j++){
                for(k=0;k<6;k++){
                    itot[i][j][k]=0.0;
                }
                for(x=R*i;x<=R*(i+1)-1;x++){
                    for(y=S*j;y<=S*(j+1)-1;y++){
                        for(ii=0;ii<32;ii++){
                            for(k=0;k<6;k++){
                                itot[i][j][k]+=(a[x][y][k].z<<ii)>>31;
                            }
                        }
                    }
                }
                total[i][j]=itot[i][j][0]+itot[i][j][1]+itot[i][j][2]+
                            itot[i][j][3]+itot[i][j][4]+itot[i][j][5];
                xi[i][j]=(itot[i][j][0]-itot[i][j][3]+((itot[i][j][1]+
                         itot[i][j][5])-(itot[i][j][2]+itot[i][j][4]))
                         /2.0)/total[i][j];
                yi[i][j]=(sqrt(3.0)/2.0*((itot[i][j][1]+itot[i][j][2])
                         -(itot[i][j][4]+itot[i][j][5])))/total[i][j];

                at[i][j]+=total[i][j]/num;
                ax[i][j]+=xi[i][j];
                ay[i][j]+=yi[i][j];

                tl+=total[i][j];
            }
        }
        for(y=0;y<=N-1;y++){
            tl+=a1[y].bool1.a+a1[y].bool1.b+a1[y].bool1.c+
                a1[y].bool1.d+a1[y].bool1.e+a1[y].bool1.f;
        }
        printf("total number=%lu\n",tl);
}
void write_on_file(h)
int h;
{
        FILE *gg;
        char sg[20],ch[20];
        int i,j;
        int p,q,pq,m,n,r,s,l;
        int exist;
        double omega,pi,den;
        
        p=P;
        q=Q;
        pq=P*Q;
        m=M*32;
        n=N;
        r=R*32;
        s=S;
        pi=PI;
        l=L;
        omega=OMEGA;
        den=(double)tl/(double)(M*32)/(double)N;
        if(h<100){
            strcpy (sg,ss);
            itoan(h,ch);
            strcat(sg,ch);
        }
        else{
            strcpy(sg,"over_100");
        }
        strcat(sg,".dat");
        gg=fopen(sg,"w");
        
        fprintf(gg,"%d %d\n",m,n);
        fprintf(gg,"%d %d\n",r,s);
        if(h==0){
            fprintf(gg,"%d\n",fermi+kk);
        }
        else{
            fprintf(gg,"%ld\n",(long)h*(lll+kk));
        }
        fprintf(gg,"%d\n",kk);
        fprintf(gg,"%d\n",pq);
        fprintf(gg,"%d %d\n",p,q);
        if(h==0){
            exist=0;
            fprintf(gg,"%d\n",exist);
            fprintf(gg,"0\n");
        }
        else{
            if(theta>=PI){
                exist=0;
                fprintf(gg,"%d\n",exist);
                fprintf(gg,"%lf\n",pi);
            }
            else{
                exist=1;
                fprintf(gg,"%d\n",exist);
                fprintf(gg,"%lf\n",theta);
            }
        }
        fprintf(gg,"%d\n",l);
        fprintf(gg,"%lu\n",tl);
        fprintf(gg,"%lf\n",omega);
        fprintf(gg,"%lf\n",den);
        for(i=0;i<P;i++) {
            for(j=0;j<Q;j++) {
                fprintf(gg,"%lf\n",at[i][j]/kk);
                fprintf(gg,"%lf %lf\n",ax[i][j]/kk,ay[i][j]/kk);
            }
        }
        fclose(gg);
        printf("h=%d\n",h);
}
/*---------------------------------------------------------------------
                     境界における衝突関数 
---------------------------------------------------------------------*/
void wing_coll_before()
{
        int i,j,k;
        double al[5],be[5];
        double l,x[5],y[5];
        double xmax,xmin,ymax,ymin;
        double itax,itan;
        double itaxx,itann,ksix,ksin;
        double we,is,dist,ans,sq;
        double u,up;
        double xo;
        unsigned ne;
        int iitan,iitax;
        int stx,sty,enx,eny;
        
        xo=M*32.0/2.0;
        al[1]=al[2]=theta;
        al[3]=al[4]=theta+PI;
        be[1]=be[3]=theta+PI/2.0;
        be[2]=be[4]=theta+3.0*PI/2.0;
        
        for(k=0;k<J;k++){
            l=SS/2.0*(2*k+1);
            x[0]=l*cos(theta);
            y[0]=l*sin(theta);
            for(i=1;i<=4;i++){
                x[i]=x[0]+SS/2.0*(cos(al[i])+cos(be[i]));
                y[i]=y[0]+SS/2.0*(sin(al[i])+sin(be[i]));
            }
            xmax=x[1];
            xmin=x[1];
            ymax=y[1];
            ymin=y[1];
            for(i=2;i<=4;i++){
                if(x[i]>xmax){
                    xmax=x[i];
                }
                if(x[i]<xmin){
                    xmin=x[i];
                }
                if(y[i]>ymax){
                    ymax=y[i];
                }
                if(y[i]<ymin){
                    ymin=y[i];
                }
            }
            itan=2.0/sqrt(3.0)*ymin;
            iitan=(int)itan;
            itann=itan-iitan;
            if(abs(iitan)%2==0){
                ksin=xo+xmin-itann/2.0;
            }
            else{
                ksin=xo+xmin+(itann-1)/2.0;
            }
            itax=2.0/sqrt(3.0)*ymax;
              iitax=(int)itax;
            itaxx=itax-iitax;
            if(abs(iitax)%2==0){
                ksix=xo+xmax-itaxx/2.0;
            }
            else{
                ksix=xo+xmax+(itaxx-1)/2.0;
            }
            stx=(int)ksin+1;
            sty=(int)itan+1;
            enx=(int)ksix;
            eny=(int)itax;
            ne=0;
            se[k]=0;
            if(stx<=0){
                stx=0;
            }
            if(enx<=0){
                enx=0;
            }
            if(sty<=1){
                sty=1;
            }
            if(eny<=1){
                eny=1;
            }
            for(i=stx;i<=enx;i++){
                for(j=sty;j<=eny;j++){
                    is=sqrt(3.0)/2.0*j;
                    if(j%2==0){
                        we=i-xo;
                    }
                    else{
                        we=i-xo+0.5;
                    }
                    dist=fabs(cos(theta)*is-sin(theta)*we);
                  sq=(we-x[0])*(we-x[0])+(is-y[0])*(is-y[0])-dist*dist;
                    if(sq<0.0){
#if PC_CHECK
                        printf("平方根のなかがマイナス！\n");
                        printf("PAUSE");
                        bdos(12,0,8);
#endif
                        sq=0.0;
                    }
                    ans=sqrt(sq);
                    if(dist<=SS/2.0&&ans<=SS/2.0){
                        mura[se[k]].px=i;
                        mura[se[k]].py=j;
                        mura[se[k]].pd=dist;
                        se[k]++;
                    }
                    ne++;
                }
            }
#if CHECK1
            printf("*** k=%d     ne=%d  se=%d ***\n",k,ne,se[k]);
#endif
            quick_sort(0,se[k]-1);
            up=l*OMEGA;
            for(i=0;i<se[k];i++){
                pxt[i][k]=mura[i].px/32;
                pyt[i][k]=mura[i].py;
                sh[i][k]=mura[i].px%32;
                am[i][k]=31-sh[i][k];
                aa[i][k].bool2.a
                         =(a[pxt[i][k]][pyt[i][k]][0].z<<sh[i][k])>>31;
                aa[i][k].bool2.b
                         =(a[pxt[i][k]][pyt[i][k]][1].z<<sh[i][k])>>31;
                aa[i][k].bool2.c
                         =(a[pxt[i][k]][pyt[i][k]][2].z<<sh[i][k])>>31;
                aa[i][k].bool2.d
                         =(a[pxt[i][k]][pyt[i][k]][3].z<<sh[i][k])>>31;
                aa[i][k].bool2.e
                         =(a[pxt[i][k]][pyt[i][k]][4].z<<sh[i][k])>>31;
                aa[i][k].bool2.f
                         =(a[pxt[i][k]][pyt[i][k]][5].z<<sh[i][k])>>31;
            }
            velocity_of_the_flow(k,&u);
#if CHECK0
            printf("---  up=%lf   initial_u=%lf  ",up,u);
#endif
            if(u==up){
                goto loop;
            }
            else if(u>up){
                for(i=0;i<se[k];i++){
                    switch(collnum)
                    {
                        case 0:     cal3_4(i,k);
                                    break;
                        case 1:     cal4_5(i,k);
                                    break;
                        case 2:     cal5_0(i,k);
                                    break;
                    }
                    velocity_of_the_flow(k,&u);
                    if(u<=up){
                        goto loop;
                    }
                }
            }
            else{
                for(i=0;i<se[k];i++){
                    switch(collnum)
                    {
                        case 0:     cal0_1(i,k);
                                    break;
                        case 1:     cal1_2(i,k);
                                    break;
                        case 2:     cal2_3(i,k);
                                    break;
                    }
                    velocity_of_the_flow(k,&u);
                    if(u>=up){
                        goto loop;
                    }
                }
            }
            printf("++++++++   u != up  forever   ++++++++\n");
loop:;
#if CHECK0
            printf(" u=%lf     << i=%d   ---\n",u,i);
#endif
        }
}
void wing_coll_after()
{
        int i,j,k;
        unsigned long lon[6];
       static unsigned long mask[32]={0x7fffffff,0xbfffffff,0xdfffffff,
                                      0xefffffff,0xf7ffffff,0xfbffffff,
                                      0xfdffffff,0xfeffffff,0xff7fffff,
                                      0xffbfffff,0xffdfffff,0xffefffff,
                                      0xfff7ffff,0xfffbffff,0xfffdffff,
                                      0xfffeffff,0xffff7fff,0xffffbfff,
                                      0xffffdfff,0xffffefff,0xfffff7ff,
                                      0xfffffbff,0xfffffdff,0xfffffeff,
                                      0xffffff7f,0xffffffbf,0xffffffdf,
                                      0xffffffef,0xfffffff7,0xfffffffb,
                                      0xfffffffd,0xfffffffe};
        
        for(k=0;k<J;k++){
            for(i=0;i<se[k];i++){
                lon[0]=aa[i][k].bool2.a;
                lon[1]=aa[i][k].bool2.b;
                lon[2]=aa[i][k].bool2.c;
                lon[3]=aa[i][k].bool2.d;
                lon[4]=aa[i][k].bool2.e;
                lon[5]=aa[i][k].bool2.f;
                for(j=0;j<6;j++){
                   a[pxt[i][k]][pyt[i][k]][j].z=
                  (a[pxt[i][k]][pyt[i][k]][j].z&mask[sh[i][k]])|
                                                    (lon[j]<<am[i][k]);
                }
            }
        }
}
void velocity_of_the_flow(k,u)
int k;
double *u;
{
        int y,i,j;
        double mt[6],total,vx,vy,v,alp,beta;
        
        for(j=0;j<=5;j++){
            mt[j]=0.0;
        }
        for(i=0;i<se[k];i++){
            mt[0]+=aa[i][k].bool2.a;
            mt[1]+=aa[i][k].bool2.b;
            mt[2]+=aa[i][k].bool2.c;
            mt[3]+=aa[i][k].bool2.d;
            mt[4]+=aa[i][k].bool2.e;
            mt[5]+=aa[i][k].bool2.f;
        }
        total=mt[0]+mt[1]+mt[2]+mt[3]+mt[4]+mt[5];
        vx=(mt[0]-mt[3]+((mt[1]+mt[5])-(mt[2]+mt[4]))/2.0)/total;
        vy=(sqrt(3.0)/2.0*((mt[1]+mt[2])-(mt[4]+mt[5])))/total;
        if(vx==0.0&&vy==0.0){
#if PC_CHECK
            printf("ｘもｙも両方ゼロ！\n");
            printf("PAUSE");
            bdos(12,0,8);
#endif
            vx=DUMMY;
            vy=DUMMY;
        }
        alp=atan2(vy,vx);
        v=sqrt(vx*vx+vy*vy);
        beta=PI/2.0-alp+theta;
        *u=v*cos(beta);
}
void t_bottom_coll()
{
        int x;
        
        for(x=0;x<=M-1;x++){
            a[x][0][1].z=a[x][0][4].z;
            a[x][0][2].z=a[x][0][5].z;
            a[x][0][4].z=(long)0x0;
            a[x][0][5].z=(long)0x0;
        }
}
void t_top_coll()
{
        int x;

        for(x=0;x<=M-1;x++){
            a[x][N-1][4].z=a[x][N-1][1].z;
            a[x][N-1][5].z=a[x][N-1][2].z;
            a[x][N-1][1].z=(long)0x0;
            a[x][N-1][2].z=(long)0x0;
        }
}
void t_right_coll()
{
        int y;

        for(y=1;y<=N-3;y+=2){
            switch(a1[y].z1)
            {
                case 0x0:       break;
                case 0x1:       a1[y].z1=0x8;
                                break;
                case 0x10:      a1[y].z1=0x2;
                                break;
                case 0x20:      a1[y].z1=0x4;
                                break;
                case 0x21:      a1[y].z1=0xc;
                                break;
                case 0x30:      a1[y].z1=0x6;
                                break;
                case 0x11:      a1[y].z1=0xa;
                                break;
                case 0x31:      a1[y].z1=0xe;
                                break;
                default:        printf("Collision Error trc\n");
            }
        }
}
void t_left_coll()
{
        int y;

        for(y=2;y<=N-2;y+=2){
            switch(a1[y].z1)
            {
                case 0x0:       break;
                case 0x8:       a1[y].z1=0x1;
                                break;
                case 0xc:       a1[y].z1=0x21;
                                break;
                case 0xa:       a1[y].z1=0x11;
                                break;
                case 0x4:       a1[y].z1=0x20;
                                break;
                case 0x2:       a1[y].z1=0x10;
                                break;
                case 0x6:       a1[y].z1=0x30;
                                break;
                case 0xe:       a1[y].z1=0x31;
                                break;
                default:        printf("Collision Error tlc\n");
            }
        }
}
void corner_coll()
{
        if(a1[0].z1==0x2)
            a1[0].z1=0x10;
        if(a1[N-1].z1==0x10)
            a1[N-1].z1=0x2;
}
void cal0_1(i,k)
int i,k;
{
        if(aa[i][k].bool2.e==1){
            if(aa[i][k].bool2.b==0){
                aa[i][k].bool2.e=0;
                aa[i][k].bool2.b=1;
            }
        }
        if(aa[i][k].bool2.f==1){
            if(aa[i][k].bool2.c==0){
                aa[i][k].bool2.f=0;
                aa[i][k].bool2.c=1;
            }
        }
        if(aa[i][k].bool2.a==1){
            if(aa[i][k].bool2.d==0){
                aa[i][k].bool2.a=0;
                aa[i][k].bool2.d=1;
            }
        }
}
void cal1_2(i,k)
int i,k;
{
        if(aa[i][k].bool2.f==1){
            if(aa[i][k].bool2.c==0){
                aa[i][k].bool2.f=0;
                aa[i][k].bool2.c=1;
            }
        }
        if(aa[i][k].bool2.a==1){
            if(aa[i][k].bool2.d==0){
                aa[i][k].bool2.a=0;
                aa[i][k].bool2.d=1;
            }
        }
        if(aa[i][k].bool2.b==1){
            if(aa[i][k].bool2.e==0){
                aa[i][k].bool2.b=0;
                aa[i][k].bool2.e=1;
            }
        }
}
void cal2_3(i,k)
int i,k;
{
        if(aa[i][k].bool2.a==1){
            if(aa[i][k].bool2.d==0){
                aa[i][k].bool2.a=0;
                aa[i][k].bool2.d=1;
            }
        }
        if(aa[i][k].bool2.b==1){
            if(aa[i][k].bool2.e==0){
                aa[i][k].bool2.b=0;
                aa[i][k].bool2.e=1;
            }
        }
        if(aa[i][k].bool2.c==1){
            if(aa[i][k].bool2.f==0){
                aa[i][k].bool2.c=0;
                aa[i][k].bool2.f=1;
            }
        }
}
void cal3_4(i,k)
int i,k;
{
        if(aa[i][k].bool2.d==1){
            if(aa[i][k].bool2.a==0){
                aa[i][k].bool2.d=0;
                aa[i][k].bool2.a=1;
            }
        }
        if(aa[i][k].bool2.b==1){
            if(aa[i][k].bool2.e==0){
                aa[i][k].bool2.b=0;
                aa[i][k].bool2.e=1;
            }
        }
        if(aa[i][k].bool2.c==1){
            if(aa[i][k].bool2.f==0){
                aa[i][k].bool2.c=0;
                aa[i][k].bool2.f=1;
            }
        }
}
void cal4_5(i,k)
int i,k;
{
        if(aa[i][k].bool2.d==1){
            if(aa[i][k].bool2.a==0){
                aa[i][k].bool2.d=0;
                aa[i][k].bool2.a=1;
            }
        }
        if(aa[i][k].bool2.e==1){
            if(aa[i][k].bool2.b==0){
                aa[i][k].bool2.e=0;
                aa[i][k].bool2.b=1;
            }
        }
        if(aa[i][k].bool2.c==1){
            if(aa[i][k].bool2.f==0){
                aa[i][k].bool2.c=0;
                aa[i][k].bool2.f=1;
            }
        }
}
void cal5_0(i,k)
int i,k;
{
        if(aa[i][k].bool2.d==1){
            if(aa[i][k].bool2.a==0){
                aa[i][k].bool2.d=0;
                aa[i][k].bool2.a=1;
            }
        }
        if(aa[i][k].bool2.e==1){
            if(aa[i][k].bool2.b==0){
                aa[i][k].bool2.e=0;
                aa[i][k].bool2.b=1;
            }
        }
        if(aa[i][k].bool2.f==1){
            if(aa[i][k].bool2.c==0){
                aa[i][k].bool2.f=0;
                aa[i][k].bool2.c=1;
            }
        }
}
void check(ck)
int ck;
{
        int i;
        int ii;
        int x,y;
        unsigned long aaa;

        aaa=0;
        for(x=0;x<=M-1;x++){
            for(y=0;y<=N-1;y++){
                for(i=0;i<=5;i++){
                    for(ii=0;ii<32;ii++){
                        aaa+=(a[x][y][i].z<<ii)>>31;
                    }
                }
            }
        }
        for(y=0;y<=N-1;y++){
            aaa+=a1[y].bool1.a+a1[y].bool1.b+a1[y].bool1.c
                 +a1[y].bool1.d+a1[y].bool1.e+a1[y].bool1.f;
        }
        printf("total.%d=%lu\n",ck,aaa);
}
/*---------------------------------------------------------------------
                  FHP-1 モデル　粒子の並進 
---------------------------------------------------------------------*/
void fhp_1()
{
        int ll;
        int i;
        int x,y;
        unsigned long ura;
        unsigned long rep[6][5];
        
        for(x=0;x<=M-1;x++){
            for(y=1;y<=N-2;y++){
                ll=urand(2);
                if(ll==1)
                    ura=~(long)0x0;
                else
                    ura=(long)0x0;

                      /***************
                      ***     A    ***
                      ***************/

                rep[0][0]=ura&a[x][y][1].z&a[x][y][4].z&~a[x][y][0].z
                            &~a[x][y][2].z&~a[x][y][3].z&~a[x][y][5].z;
                rep[0][1]=~ura&a[x][y][2].z&a[x][y][5].z&
                            ~a[x][y][0].z&~ a[x][y][1].z&~a[x][y][3].z
                            &~a[x][y][4].z;
                rep[0][2]=a[x][y][0].z&a[x][y][3].z&~a[x][y][1].z&
                            ~a[x][y][2].z&~a[x][y][4].z&~a[x][y][5].z;
                rep[0][3]=a[x][y][1].z&a[x][y][3].z&a[x][y][5].z
                            &~a[x][y][0].z&~a[x][y][2].z&~a[x][y][4].z;
                rep[0][4]=a[x][y][0].z &a[x][y][2].z & a[x][y][4].z
                            &~a[x][y][1].z&~a[x][y][3].z&~a[x][y][5].z;

                      /***************
                      ***     B    ***
                      ***************/

                rep[1][0]=ura&a[x][y][2].z&a[x][y][5].z&~a[x][y][1].z
                            &~a[x][y][3].z&~a[x][y][4].z&~a[x][y][0].z;
                rep[1][1]=~ura&a[x][y][3].z&a[x][y][0].z&
                            ~a[x][y][1].z&~a[x][y][2].z&~a[x][y][4].z
                            &~a[x][y][5].z;
                rep[1][2]=a[x][y][1].z&a[x][y][4].z&~a[x][y][2].z
                            &~a[x][y][3].z&~a[x][y][5].z&~a[x][y][0].z;
                rep[1][3]=a[x][y][2].z&a[x][y][4].z&a[x][y][0].z
                            &~a[x][y][1].z&~a[x][y][3].z&~a[x][y][5].z;
                rep[1][4]=a[x][y][1].z&a[x][y][3].z&a[x][y][5].z
                            &~a[x][y][2].z&~a[x][y][4].z&~a[x][y][0].z;

                      /***************
                      ***     C    ***
                      ***************/

                rep[2][0]=ura&a[x][y][3].z&a[x][y][0].z&~a[x][y][2].z
                            &~a[x][y][4].z&~a[x][y][5].z&~a[x][y][1].z;
                rep[2][1]=~ura&a[x][y][4].z&a[x][y][1].z
                            &~a[x][y][2].z&~a[x][y][3].z&~a[x][y][5].z
                            &~a[x][y][0].z;
                rep[2][2]=a[x][y][2].z&a[x][y][5].z&~a[x][y][3].z
                            &~a[x][y][4].z&~a[x][y][0].z&~a[x][y][1].z;
                rep[2][3]=a[x][y][3].z&a[x][y][5].z&a[x][y][1].z
                            &~a[x][y][2].z&~a[x][y][4].z&~a[x][y][0].z;
                rep[2][4]=a[x][y][2].z&a[x][y][4].z&a[x][y][0].z
                            &~a[x][y][3].z&~a[x][y][5].z&~a[x][y][1].z;

                      /***************
                      ***     D    ***
                      ***************/

                rep[3][0]=ura&a[x][y][4].z&a[x][y][1].z&~a[x][y][3].z
                            &~a[x][y][5].z&~a[x][y][0].z&~a[x][y][2].z;
                rep[3][1]=~ura&a[x][y][5].z&a[x][y][2].z
                            &~a[x][y][3].z&~a[x][y][4].z&~a[x][y][0].z
                            &~a[x][y][1].z;
                rep[3][2]=a[x][y][3].z&a[x][y][0].z&~a[x][y][4].z
                            &~a[x][y][5].z&~a[x][y][1].z&~a[x][y][2].z;
                rep[3][3]=a[x][y][4].z&a[x][y][0].z&a[x][y][2].z
                            &~a[x][y][3].z&~a[x][y][5].z&~a[x][y][1].z;
                rep[3][4]=a[x][y][3].z&a[x][y][5].z&a[x][y][1].z
                            &~a[x][y][4].z&~a[x][y][0].z&~a[x][y][2].z;

                      /***************
                      ***     E    ***
                      ***************/

                rep[4][0]=ura&a[x][y][5].z&a[x][y][2].z&~a[x][y][4].z
                            &~a[x][y][0].z&~a[x][y][1].z&~a[x][y][3].z;
                rep[4][1]=~ura&a[x][y][0].z&a[x][y][3].z
                            &~a[x][y][4].z&~a[x][y][5].z&~a[x][y][1].z
                            &~a[x][y][2].z;
                rep[4][2]=a[x][y][4].z&a[x][y][1].z&~a[x][y][5].z
                            &~a[x][y][0].z&~a[x][y][2].z&~a[x][y][3].z;
                rep[4][3]=a[x][y][5].z&a[x][y][1].z&a[x][y][3].z
                            &~a[x][y][4].z&~a[x][y][0].z&~a[x][y][2].z;
                rep[4][4]=a[x][y][4].z&a[x][y][0].z&a[x][y][2].z
                            &~a[x][y][5].z&~a[x][y][1].z&~a[x][y][3].z;

                      /***************
                      ***     F    ***
                      ***************/

                rep[5][0]=ura&a[x][y][0].z&a[x][y][3].z&~a[x][y][5].z
                            &~a[x][y][1].z&~a[x][y][2].z&~a[x][y][4].z;
                rep[5][1]=~ura&a[x][y][1].z&a[x][y][4].z
                            &~a[x][y][5].z&~a[x][y][0].z&~a[x][y][2].z
                            &~a[x][y][3].z;
                rep[5][2]=a[x][y][5].z&a[x][y][2].z&~a[x][y][0].z
                            &~a[x][y][1].z&~a[x][y][3].z&~a[x][y][4].z;
                rep[5][3]=a[x][y][0].z&a[x][y][2].z&a[x][y][4].z
                            &~a[x][y][5].z&~a[x][y][1].z&~a[x][y][3].z;
                rep[5][4]=a[x][y][5].z&a[x][y][1].z&a[x][y][3].z
                            &~a[x][y][0].z&~a[x][y][2].z&~a[x][y][4].z;

                for(i=0;i<=5;i++){
                    a[x][y][i].z=a[x][y][i].z^rep[i][0]^rep[i][1]
                                 ^rep[i][2]^rep[i][3]^rep[i][4];
                }
            }
        }
}
void inner_advance()
{
        int x,y;
        unsigned char D[N/2];

        for(y=1;y<=N/2-1;y++){
            D[y]=a[M-1][2*y][0].bool.a32;
        }
                      /***************
                      ***     A    ***
                      ***************/

        for(y=1;y<=N-3;y+=2){
            a1[y].bool1.a=a[M-1][y][0].bool.a32;
        }
        for(y=1;y<=N-2;y++){
            for(x=M-1;x>=1;x--){
                a[x][y][0].z=a[x][y][0].z>>1;
                a[x][y][0].bool.a1=a[x-1][y][0].bool.a32;
            }
        }
        for(y=1;y<=N-3;y+=2){
            a[0][y][0].z=a[0][y][0].z>>1;
            a[0][y][0].bool.a1=a[0][y][3].bool.a1;
        }
        for(y=2;y<=N-2;y+=2){
            a[0][y][0].z=a[0][y][0].z>>1;
            a[0][y][0].bool.a1=a1[y].bool1.a;
        }
        for(y=2;y<=N-2;y+=2){
            a1[y].bool1.a=0x0;
        }
                      /***************
                      ***     B    ***
                      ***************/
        for(y=1;y<=N-1;y+=2){
            a1[y].bool1.b=a[M-1][y-1][1].bool.a32;
        }
        for(x=M-1;x>=1;x--){
            for(y=N-1;y>=3;y-=2){
                a[x][y][1].z=a[x][y-1][1].z>>1;
                a[x][y][1].bool.a1=a[x-1][y-1][1].bool.a32;
                a[x][y-1][1].z=a[x][y-2][1].z;
            }
            a[x][1][1].z=a[x][0][1].z>>1;
            a[x][1][1].bool.a1=a[x-1][0][1].bool.a32;
            a[x][0][1].z=(long)0x0;
        }
        for(y=N-1;y>=3;y-=2){
            a[0][y][1].z=a[0][y-1][1].z>>1;
            a[0][y][1].bool.a1=a1[y-1].bool1.b;
            a[0][y-1][1].z=a[0][y-2][1].z;
        }
        a[0][1][1].z=a[0][0][1].z>>1;
        a[0][1][1].bool.a1=a1[0].bool1.b;
        a[0][0][1].z=(long)0x0;
        for(y=0;y<=N-2;y+=2){
            a1[y].bool1.b=0x0;
        }
                      /***************
                      ***     C    ***
                      ***************/
        for(y=N-2;y>=2;y-=2){
            a1[y].bool1.c=a[0][y-1][2].bool.a1;
        }
        for(x=0;x<=M-2;x++){
            for(y=N-1;y>=3;y-=2){
                a[x][y][2].z=a[x][y-1][2].z;
                a[x][y-1][2].z=a[x][y-2][2].z<<1;
                a[x][y-1][2].bool.a32=a[x+1][y-2][2].bool.a1;
            }
            a[x][1][2].z=a[x][0][2].z;
            a[x][0][2].z=(long)0x0;
        }
        for(y=N-1;y>=3;y-=2){
            a[M-1][y][2].z=a[M-1][y-1][2].z;
            a[M-1][y-1][2].z=a[M-1][y-2][2].z<<1;
            a[M-1][y-1][2].bool.a32=a1[y-2].bool1.c;
        }
        a[M-1][1][2].z=a[M-1][0][2].z;
        a[M-1][0][2].z=(long)0x0;
        for(y=N-1;y>=1;y-=2){
            a1[y].bool1.c=0x0;
        }
                      /***************
                      ***     D    ***
                      ***************/
        for(y=2;y<=N-2;y+=2){
            a1[y].bool1.d=a[0][y][3].bool.a1;
        }
        for(y=1;y<=N-2;y++){
            for(x=0;x<=M-2;x++){
                a[x][y][3].z=a[x][y][3].z<<1;
                a[x][y][3].bool.a32=a[x+1][y][3].bool.a1;
            }
        }
        for(y=1;y<=N-3;y+=2){
            a[M-1][y][3].z=a[M-1][y][3].z<<1;
            a[M-1][y][3].bool.a32=a1[y].bool1.d;
        }
        for(y=2;y<=N-2;y+=2){
            a[M-1][y][3].z=a[M-1][y][3].z<<1;
            a[M-1][y][3].bool.a32=D[y/2];
        }
        for(y=1;y<=N-1;y+=2){
            a1[y].bool1.d=0x0;
        }
                      /***************
                      ***     E    ***
                      ***************/
        for(y=0;y<=N-2;y+=2){
            a1[y].bool1.e=a[0][y+1][4].bool.a1;
        }
        for(x=0;x<=M-2;x++){
            for(y=0;y<=N-4;y+=2){
                a[x][y][4].z=a[x][y+1][4].z<<1;
                a[x][y][4].bool.a32=a[x+1][y+1][4].bool.a1;
                a[x][y+1][4].z=a[x][y+2][4].z;
            }
            a[x][N-2][4].z=a[x][N-1][4].z<<1;
            a[x][N-2][4].bool.a32=a[x+1][N-1][4].bool.a1;
            a[x][N-1][4].z=(long)0x0;
        }
        for(y=0;y<=N-4;y+=2){
            a[M-1][y][4].z=a[M-1][y+1][4].z<<1;
            a[M-1][y][4].bool.a32=a1[y+1].bool1.e;
            a[M-1][y+1][4].z=a[M-1][y+2][4].z;
        }
        a[M-1][N-2][4].z=a[M-1][N-1][4].z<<1;
        a[M-1][N-2][4].bool.a32=a1[N-1].bool1.e;
        a[M-1][N-1][4].z=(long)0x0;
        for(y=1;y<=N-1;y+=2){
                a1[y].bool1.e=0x0;
        }
                      /***************
                      ***     F    ***
                      ***************/
        for(y=1;y<=N-3;y+=2){
            a1[y].bool1.f=a[M-1][y+1][5].bool.a32;
        }
        for(x=M-1;x>=1;x--){
            for(y=0;y<=N-4;y+=2){
                a[x][y][5].z=a[x][y+1][5].z;
                a[x][y+1][5].z=a[x][y+2][5].z>>1;
                a[x][y+1][5].bool.a1=a[x-1][y+2][5].bool.a32;
            }
            a[x][N-2][5].z=a[x][N-1][5].z;
            a[x][N-1][5].z=(long)0x0;
        }
        for(y=0;y<=N-4;y+=2){
            a[0][y][5].z=a[0][y+1][5].z;
            a[0][y+1][5].z=a[0][y+2][5].z>>1;
            a[0][y+1][5].bool.a1=a1[y+2].bool1.f;
        }
        a[0][N-2][5].z=a[0][N-1][5].z;
        a[0][N-1][5].z=(long)0x0;
        for(y=0;y<=N-2;y+=2){
            a1[y].bool1.f=0x0;
        }
}
/*---------------------------------------------------------------------
                     オプション
---------------------------------------------------------------------*/
int urand(n)
int n;
{
        int x;
        static unsigned long ir2=12345;

        ir2=(LAMBDA*ir2+MU)%LL;
        x=ir2%n;
        return x;
}
void itoan(x,c)
int x;
char *c;
{
        int i,j;

        i=x/10;
        j=x%10;
        *c=48+i;
        *(c+1)=48+j;
        *(c+2)=NULL;
}
void quick_sort(l0,h0)
int l0;
int h0;
{
    int l,h;
    double m;
    struct {int aa; int bb; double pp;}mr;
    
    if(l0<h0){
        l=l0;
        h=h0;
        m=mura[(l+h)>>1].pd;
        do{
            while(mura[l].pd<m){
                l++;
            }
            while(mura[h].pd>m){
                h--;
            }
            if(l<=h){
                mr.aa=mura[l].px;
                mr.bb=mura[l].py;
                mr.pp=mura[l].pd;
                mura[l].px=mura[h].px;
                mura[l].py=mura[h].py;
                mura[l].pd=mura[h].pd;
                mura[h].px=mr.aa;
                mura[h].py=mr.bb;
                mura[h].pd=mr.pp;
                l++;
                h--;
            }
        }while(l<=h);
        if(h-l0<h0-1){
            quick_sort(l0,h);
            quick_sort(l,h0);
        }
        else{
            quick_sort(l,h0);
            quick_sort(l0,h);
        }
    }
}

