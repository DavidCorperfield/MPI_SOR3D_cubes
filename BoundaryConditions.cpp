/*
 *  BoundaryConditions.cpp
 *  Created by Jeremy Riousset on 6/26/08.
 */

#include "BoundaryConditions.h"
/**************************************************************************************/
void BC::Apply(int BBCtype, CMatrix3D& pphi, CMatrix3D& llocal_rho)
{
	double start, finish, runtime(0);
	double xx,yy,zz,xp,yp,zp, dV(Var::d.x*Var::d.y*Var::d.z), k(1/(4*M_PI*eps0));
	
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
	
	/**********************************************************************************/
	/* This function modifies the field at boundaries else than z = 0, however, once  */
	/* fixed, this potential is not modified.										  */
	/* If BBCtype =																	  */
	/*		0. ``Tin Can Model": the potential on the boundaries is fixed at 0		  */
	/*		1. ``Open BC Model": PEC ground, open BC everywhere on the sides		  */
	/*      2. ``Capacitor Model": PEC ground and electrosphere, open side boundaries */
	/*      3. ``Free Space Model": open side, bottom and top boundaries			  */
	/**********************************************************************************/
	
	if(BBCtype!=0 && BBCtype!=1 && BBCtype!=2 && BBCtype!=3)
	{
		if(MPI_Var::world_rank==MPI_Var::root)	printf("Incorrect Type of Boundary Conditions\n");
		exit(4);
	}
	else if(BBCtype==0)
	{
		// No change needed //
	}
	else if(BBCtype==1 || BBCtype==2 || BBCtype==3)
	{
		// To reduce the time of computation, only the source points exceeding xx% of //
		// the maximal charge density -in absolute value- will be considered.		  //
		double rhoMax = 0;
		double local_rhoMax = 0;
		int		M(25);								// Account for M ground images and M ionospheric images
		double	z_GndImg = 0;						// Altitude of ground images
		double	z_IonImg = 0;						// Altitude of iono/electrosphere images
		double	z_Ion    = (Var::N.z-1)*Var::d.z;	// Altitude coordinate of the iono/electrosphere
		
		for(int ip=MPI_Var::is ; ip<=MPI_Var::ie ; ip++) for(int jp=MPI_Var::js ; jp<=MPI_Var::je ; jp++) for(int kp=MPI_Var::ks ; kp<=MPI_Var::ke ; kp++)
			if(fabs(llocal_rho[ip][jp][kp]) > local_rhoMax) local_rhoMax = fabs(llocal_rho[ip][jp][kp]);
	
		MPI_Allreduce(&local_rhoMax, &rhoMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		// We neglect the ambient Laplacian field on Earth (100 V/m = 1e-3kV/cm)	  //
		// The potential due to the ambient Laplacian field VL = 0.					  //
		// To take into account this potential, VL = 100 * z (in V)					  //
		// NB: subscript "s" refers to the source, which volume we are integrating on.//
		//	   Hence "s" refers to variable of integration and no subscript refers to //
		//     the position of the boundary.										  //
		double	Eambient = 0;
		double	VL;
		
		// Initiate all boundary using the ambient potential //
		
		for(int ii=1 ; ii<Var::N.x-1 ; ii++) for(int jj=1 ; jj<Var::N.y-1 ; jj++) 
		{
			zz = 0*Var::d.z; 
			VL = Eambient * zz;
			MPI_Var::bot_plane[ii-1][jj-1] =	0*VL;

			zz = (Var::N.z-1)*Var::d.z; 
			VL = Eambient * zz;
			MPI_Var::bot_plane[ii-1][jj-1] =	0*VL;
		}
		for(int ii=1 ; ii<Var::N.x-1 ; ii++) for(int kk=0 ; kk<Var::N.z ; kk++) 
		{
			zz = kk*Var::d.z; 
			VL = Eambient * zz;
			MPI_Var::left_plane[ii-1][kk]	=	VL;
			MPI_Var::right_plane[ii-1][kk]	=	VL;
		}
		for(int jj=0 ; jj<Var::N.y ; jj++) for(int kk=0 ; kk<Var::N.z ; kk++) 
		{
			zz = kk*Var::d.z; 
			VL = Eambient * zz;
			MPI_Var::back_plane[jj][kk]		=	VL;
			MPI_Var::front_plane[jj][kk]	=	VL;
		}
		
		// Calculate the contribution of the charge in the domain to the potential at the boundaries //
//		if(MPI_Var::world_rank==MPI_Var::root) cout<<Var::N.x<<" "<<Var::N.y<<" "<<Var::N.z<<" "<<Var::local_N.x<<" "<<Var::local_N.y<<" "<<Var::local_N.z<<" "<<MPI_Var::is<<" "<<MPI_Var::ie<<endl;
		for(int ip=MPI_Var::is ; ip<=MPI_Var::ie ; ip++) for(int jp=MPI_Var::js ; jp<=MPI_Var::je ; jp++) for(int kp=MPI_Var::ks ; kp<=MPI_Var::ke ; kp++)
		{
			if(fabs(llocal_rho[ip][jp][kp]) > .01*rhoMax)
			{
				xp = (MPI_Var::x_rank * (Var::local_N.x-2) + ip)*Var::d.x;
				yp = (MPI_Var::y_rank * (Var::local_N.y-2) + jp)*Var::d.y;
				zp = (MPI_Var::z_rank * (Var::local_N.z-2) + kp)*Var::d.z;
				
				for(int ii=1 ; ii<Var::N.x-1 ; ii++) for(int jj=1 ; jj<Var::N.y-1 ; jj++) 
				{
					xx = (ii-1)*Var::d.x;
					yy = jj*Var::d.y;
					
					zz = 0*Var::d.z;
					if(!(xx == xp && yy == yp && zz == zp))
					{	
						if(BBCtype==1) 
							MPI_Var::bot_plane[ii-1][jj-1] +=	k*dV * llocal_rho[ip][jp][kp]*(1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz-zp,2 )) - 1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz+zp,2 )));
						if(BBCtype==2) 
						{
							/*z_GndImg = zp; // Altitude of ground images			   //
							z_IonImg = zp; // Altitude of iono/electrosphere images //
							MPI_Var::bot_plane[ii-1][jj-1] +=	k*dV * llocal_rho[ip][jp][kp]/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz-zp,2 ));
							for(int mm=1; mm<=M; mm++)
							{
								z_GndImg = z_GndImg - (		mm%2*2*zp	+	(mm-1)%2*2*(z_Ion-zp) );
								z_IonImg = z_IonImg + ( (mm-1)%2*2*zp	+		mm%2*2*(z_Ion-zp) );
								MPI_Var::bot_plane[ii-1][jj-1] +=	k*dV * llocal_rho[ip][jp][kp]* pow(-1.0,mm)*
									(1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz - z_GndImg,2 )) +	// Ground Images
									 1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz - z_IonImg,2 )) ); // Ionosphere Images
							}*/
						}
						if(BBCtype==3) 
							MPI_Var::bot_plane[ii-1][jj-1] +=	k*dV * llocal_rho[ip][jp][kp]/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz-zp,2 ));
					}
					
					zz = (Var::N.z-1)*Var::d.z;
					if(!(xx == xp && yy == yp && zz == zp)) 
					{	
						if(BBCtype==1) 
							MPI_Var::top_plane[ii-1][jj-1] +=	k*dV * llocal_rho[ip][jp][kp]*(1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz-zp,2 )) - 1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz+zp,2 )));
						if(BBCtype==2) 
						{
							/*z_GndImg = zp; // Altitude of ground images			   //
							z_IonImg = zp; // Altitude of iono/electrosphere images //
							MPI_Var::top_plane[ii-1][jj-1] +=	k*dV * llocal_rho[ip][jp][kp]/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz-zp,2 ));
							for(int mm=1; mm<=M; mm++)
							{
								z_GndImg = z_GndImg - (		mm%2*2*zp	+	(mm-1)%2*2*(z_Ion-zp) );
								z_IonImg = z_IonImg + ( (mm-1)%2*2*zp	+		mm%2*2*(z_Ion-zp) );
								MPI_Var::top_plane[ii-1][jj-1] +=	k*dV * llocal_rho[ip][jp][kp]* pow(-1.0,mm)*
									(1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz - z_GndImg,2 )) +	// Ground Images
									 1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz - z_IonImg,2 )) ); // Ionosphere Images
							}*/
						}
						if(BBCtype==3) 
							MPI_Var::top_plane[ii-1][jj-1] +=	k*dV * llocal_rho[ip][jp][kp]/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz-zp,2 ));
					}
				}
				for(int ii=1 ; ii<Var::N.x-1 ; ii++) for(int kk=0 ; kk<Var::N.z ; kk++) 
				{
					xx = ii*Var::d.x;
					zz = kk*Var::d.z;
					
					yy = 0*Var::d.y; 
					if(!(xx == xp && yy == yp && zz == zp)) 
					{	
						if(BBCtype==1) 
							MPI_Var::left_plane[ii-1][kk] +=	k*dV * llocal_rho[ip][jp][kp]*(1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz-zp,2 )) - 1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz+zp,2 )));
						if(BBCtype==2) 
						{
							z_GndImg = zp; // Altitude of ground images			   //
							z_IonImg = zp; // Altitude of iono/electrosphere images //
							MPI_Var::left_plane[ii-1][kk] +=	k*dV * llocal_rho[ip][jp][kp]/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz-zp,2 ));
							for(int mm=1; mm<=M; mm++)
							{
								z_GndImg = z_GndImg - (		mm%2*2*zp	+	(mm-1)%2*2*(z_Ion-zp) );
								z_IonImg = z_IonImg + ( (mm-1)%2*2*zp	+		mm%2*2*(z_Ion-zp) );
								MPI_Var::left_plane[ii-1][kk] +=	k*dV * llocal_rho[ip][jp][kp]* pow(-1.0,mm)*
									(1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz - z_GndImg,2 )) +	// Ground Images
									 1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz - z_IonImg,2 )) ); // Ionosphere Images
							}
						}
						if(BBCtype==3) 
							MPI_Var::left_plane[ii-1][kk] +=	k*dV * llocal_rho[ip][jp][kp]/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz-zp,2 ));
					}
					yy = (Var::N.y-1)*Var::d.y; 
					if(!(xx == xp && yy == yp && zz == zp)) 
					{	
						if(BBCtype==1) 
							MPI_Var::right_plane[ii-1][kk] +=	k*dV * llocal_rho[ip][jp][kp]*(1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz-zp,2 )) - 1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz+zp,2 )));
						if(BBCtype==2) 
						{
							z_GndImg = zp; // Altitude of ground images			   //
							z_IonImg = zp; // Altitude of iono/electrosphere images //
							MPI_Var::right_plane[ii-1][kk] +=	k*dV * llocal_rho[ip][jp][kp]/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz-zp,2 ));
							for(int mm=1; mm<=M; mm++)
							{
								z_GndImg = z_GndImg - (		mm%2*2*zp	+	(mm-1)%2*2*(z_Ion-zp) );
								z_IonImg = z_IonImg + ( (mm-1)%2*2*zp	+		mm%2*2*(z_Ion-zp) );
								MPI_Var::right_plane[ii-1][kk] +=	k*dV * llocal_rho[ip][jp][kp]* pow(-1.0,mm)*
									(1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz - z_GndImg,2 )) +	// Ground Images
									 1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz - z_IonImg,2 )) ); // Ionosphere Images
							}
						}
						if(BBCtype==3) 
							MPI_Var::right_plane[ii-1][kk] +=	k*dV * llocal_rho[ip][jp][kp]/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz-zp,2 ));
					}
				}
				for(int jj=0 ; jj<Var::N.y ; jj++) for(int kk=0 ; kk<Var::N.z ; kk++) 
				{
					yy = jj*Var::d.y;
					zz = kk*Var::d.z;
					
					xx = 0*Var::d.x; 
					if(!(xx == xp && yy == yp && zz == zp)) 
					{	
						if(BBCtype==1) 
							MPI_Var::back_plane[jj][kk] +=	k*dV * llocal_rho[ip][jp][kp]*(1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz-zp,2 )) - 1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz+zp,2 )));
						if(BBCtype==2) 
						{
							z_GndImg = zp; // Altitude of ground images			   //
							z_IonImg = zp; // Altitude of iono/electrosphere images //
							MPI_Var::back_plane[jj][kk] +=	k*dV * llocal_rho[ip][jp][kp]/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz-zp,2 ));
							for(int mm=1; mm<=M; mm++)
							{
								z_GndImg = z_GndImg - (		mm%2*2*zp	+	(mm-1)%2*2*(z_Ion-zp) );
								z_IonImg = z_IonImg + ( (mm-1)%2*2*zp	+		mm%2*2*(z_Ion-zp) );
								MPI_Var::back_plane[jj][kk] +=	k*dV * llocal_rho[ip][jp][kp]* pow(-1.0,mm)*
									(1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz - z_GndImg,2 )) +	// Ground Images
									 1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz - z_IonImg,2 )) ); // Ionosphere Images
							}
						}
						if(BBCtype==3) 
							MPI_Var::back_plane[jj][kk] +=	k*dV * llocal_rho[ip][jp][kp]/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz-zp,2 ));
					}
					xx = (Var::N.x-1)*Var::d.x; 
					if(!(xx == xp && yy == yp && zz == zp)) 
					{	
						if(BBCtype==1) 
							MPI_Var::front_plane[jj][kk] +=	k*dV * llocal_rho[ip][jp][kp]*(1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz-zp,2 )) - 1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz+zp,2 )));
						if(BBCtype==2) 
						{
							z_GndImg = zp; // Altitude of ground images			   //
							z_IonImg = zp; // Altitude of iono/electrosphere images //
							MPI_Var::front_plane[jj][kk] +=	k*dV * llocal_rho[ip][jp][kp]/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz-zp,2 ));
							for(int mm=1; mm<=M; mm++)
							{
								z_GndImg = z_GndImg - (		mm%2*2*zp	+	(mm-1)%2*2*(z_Ion-zp) );
								z_IonImg = z_IonImg + ( (mm-1)%2*2*zp	+		mm%2*2*(z_Ion-zp) );
								MPI_Var::front_plane[jj][kk] +=	k*dV * llocal_rho[ip][jp][kp]* pow(-1.0,mm)*
									(1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz - z_GndImg,2 )) +	// Ground Images
									 1/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz - z_IonImg,2 )) ); // Ionosphere Images
							}
						}
						if(BBCtype==3) 
							MPI_Var::front_plane[jj][kk] +=	k*dV * llocal_rho[ip][jp][kp]/sqrt(pow( xx-xp,2 ) + pow( yy-yp,2 ) + pow( zz-zp,2 ));
					}
				}
			}
		}
	}
	MPI_foo::Reduce(pphi);
	MPI_Barrier(MPI_COMM_WORLD);
	finish	 = MPI_Wtime();
	runtime	+= (finish - start);
	
	if(MPI_Var::world_rank==MPI_Var::root)	printf("Run time for BC  solver: %e s\n",runtime);
}
