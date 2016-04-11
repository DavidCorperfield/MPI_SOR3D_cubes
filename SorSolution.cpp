/*
 *  SorSolution.cpp
 *  Created by Jeremy Riousset on 11/19/07.
 */

#include "SorSolution.h"

SorSolution::SorSolution(CMatrix3D& pphi, double eepsilon,int MMaxStep, ResGrid dd,SizeGrid NN, Charge& CC,		const	CMatrix3D& UUn)
{SorSolution::init(pphi, eepsilon,MMaxStep, dd,NN, CC, UUn);};

SorSolution::SorSolution(CMatrix3D& pphi, double eepsilon,int MMaxStep, ResGrid dd,SizeGrid NN, Potential& PP,			CMatrix3D& UUn)
{SorSolution::init(pphi, eepsilon,MMaxStep, dd,NN, PP, UUn);};

void SorSolution::init(CMatrix3D& local_phi, double eepsilon, int MMaxStep, ResGrid dd, SizeGrid local_N, Charge& local_C, const CMatrix3D& local_Un)
{
	double	local_ErrDen;
	double	gghat = -2*(1/pow(dd.x,2)+1/pow(dd.y,2)+1/pow(dd.z,2));
	type	= ChargeDistribution;
	epsilon	= eepsilon;
	MaxStep	= MMaxStep;
	
	a				= 1 / pow(dd.x,2) / gghat;
    b				= 1 / pow(dd.x,2) / gghat;
    c				= 1 / pow(dd.y,2) / gghat;
    d				= 1 / pow(dd.y,2) / gghat;
    e				= 1 / pow(dd.z,2) / gghat;
    f				= 1 / pow(dd.z,2) / gghat;
    g				= 1 ;
	ErrDen			= 0 ;
	
	h				= local_C.rho;
	local_ErrDen	= 0;
	for(int  kk=0 ; kk<local_N.z ; kk++) for(int  jj=0 ; jj<local_N.y ; jj++) for(int  ii=0 ; ii<local_N.x ; ii++)
	{
		if ( ii!=0 && ii!=local_N.x-1 &&  jj!=0 &&  jj!=local_N.y-1 &&  kk!=0 && kk!=local_N.z-1)
		{h[ii][jj][kk] /= (-eps0*gghat);}
		else
		{h[ii][jj][kk]	= 0;}
		local_ErrDen += (local_Un[ii][jj][kk]==0)*pow(h[ii][jj][kk],2);
	}
	MPI_Allreduce(&local_ErrDen, &ErrDen, 1,	MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}; // Init Charge

void SorSolution::init(CMatrix3D& local_phi, double eepsilon, int MMaxStep, ResGrid dd, SizeGrid local_N, Potential& local_P,		CMatrix3D& local_Un)
{
	double local_ErrDen;
	double gghat	= -2*(1/pow(dd.x,2)+1/pow(dd.y,2)+1/pow(dd.z,2));
	type			= PotentialDistribution;
	epsilon			= eepsilon;
	MaxStep			= MMaxStep;
	
	a				= 1 / pow(dd.x,2) / gghat;
    b				= 1 / pow(dd.x,2) / gghat;
    c				= 1 / pow(dd.y,2) / gghat;
    d				= 1 / pow(dd.y,2) / gghat;
    e				= 1 / pow(dd.z,2) / gghat;
    f				= 1 / pow(dd.z,2) / gghat;
    g				= 1 ;
	ErrDen			= 0 ;
	local_Un		= local_P.Un;
	
	if(local_P.getEquiPotential()==true)
	{
		double VVo		= local_P.getVo();	
		for(int ii = 0 ; ii<local_N.x ; ii++) for(int jj = 0 ; jj<local_N.y ; jj++) for(int kk = 0 ; kk<local_N.z ; kk++)
			if(local_P.Un[ii][jj][kk] !=0) 
			{
				if ( ii!=0 && ii!=local_N.x-1 &&  jj!=0 &&  jj!=local_N.y-1 &&  kk!=0 && kk!=local_N.z-1)
				{local_phi[ii][jj][kk]	= VVo;}
				local_ErrDen	   += pow(VVo,2);
			};
	}
	else if(local_P.getEquiPotential()==false)
		for(int ii = 0 ; ii<local_N.x ; ii++) for(int jj = 0 ; jj<local_N.y ; jj++) for(int kk = 0 ; kk<local_N.z ; kk++)
		{
			if(local_P.Un[ii][jj][kk]!=0)
			{
				if ( ii!=0 && ii!=local_N.x-1 &&  jj!=0 &&  jj!=local_N.y-1 &&  kk!=0 && kk!=local_N.z-1)
				{local_phi[ii][jj][kk]	 = local_P.rho[ii][jj][kk];}
				local_ErrDen    	+= pow(local_P.rho[ii][jj][kk],2);
			};
		};
	MPI_Allreduce(&local_ErrDen, &ErrDen, 1,	MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}; // Init Potential

void SorSolution::Update_GhostPlanes(CMatrix3D& local_MM, SizeGrid local_NN, PointsColor CC)
{	
	if(CC == red)
	{
		//SEND/RECV FRONT ROW//
		MPI_Send(&local_MM.pElems[(local_NN.x-2)*local_NN.y*local_NN.z],	1,	MPI_Var::local_x_plane_blk,	MPI_Var::next_x_rank,	MPI_Var::tag,	MPI_Var::x_comm );
		MPI_Recv(&local_MM.pElems[0],										1,	MPI_Var::local_x_plane_blk,	MPI_Var::prev_x_rank,	MPI_Var::tag,	MPI_Var::x_comm, &MPI_Var::status);
		
		//SEND/RECV BACK ROW//
		MPI_Send(&local_MM.pElems[local_NN.y*local_NN.z],					1,	MPI_Var::local_x_plane_red,	MPI_Var::prev_x_rank,	MPI_Var::tag,	MPI_Var::x_comm );
		MPI_Recv(&local_MM.pElems[(local_NN.x-1)*local_NN.y*local_NN.z],	1,	MPI_Var::local_x_plane_red,	MPI_Var::next_x_rank,	MPI_Var::tag,	MPI_Var::x_comm, &MPI_Var::status);
		
		//SEND/RECV LEFT ROW//
		MPI_Send(&local_MM.pElems[(local_NN.y-2)*local_NN.z],				1,	MPI_Var::local_y_plane_blk,	MPI_Var::next_y_rank,	MPI_Var::tag,	MPI_Var::y_comm );
		MPI_Recv(&local_MM.pElems[0],										1,	MPI_Var::local_y_plane_blk,	MPI_Var::prev_y_rank,	MPI_Var::tag,	MPI_Var::y_comm, &MPI_Var::status);
		
		//SEND/RECV RIGHT ROW//
		MPI_Send(&local_MM.pElems[local_NN.z],								1,	MPI_Var::local_y_plane_red,	MPI_Var::prev_y_rank,	MPI_Var::tag,	MPI_Var::y_comm );
		MPI_Recv(&local_MM.pElems[(local_NN.y-1)*local_NN.z],				1,	MPI_Var::local_y_plane_red,	MPI_Var::next_y_rank,	MPI_Var::tag,	MPI_Var::y_comm, &MPI_Var::status);
		
		//SEND/RECV TOP ROW//
		MPI_Send(&local_MM.pElems[local_NN.z-2],							1,	MPI_Var::local_z_plane_blk, MPI_Var::next_z_rank,	MPI_Var::tag,	MPI_Var::z_comm );
		MPI_Recv(&local_MM.pElems[0],										1,	MPI_Var::local_z_plane_blk,	MPI_Var::prev_z_rank,	MPI_Var::tag,	MPI_Var::z_comm, &MPI_Var::status);
		
		//SEND/RECV BOTTOM ROW//
		MPI_Send(&local_MM.pElems[1],										1,	MPI_Var::local_z_plane_red,	MPI_Var::prev_z_rank,	MPI_Var::tag,	MPI_Var::z_comm );
		MPI_Recv(&local_MM.pElems[local_NN.z-1],							1,	MPI_Var::local_z_plane_red,	MPI_Var::next_z_rank,	MPI_Var::tag,	MPI_Var::z_comm, &MPI_Var::status);
	}
	if(CC == black)
	{
		//SEND/RECV FRONT ROW//
		MPI_Send(&local_MM.pElems[(local_NN.x-2)*local_NN.y*local_NN.z],	1,	MPI_Var::local_x_plane_red,	MPI_Var::next_x_rank,	MPI_Var::tag,	MPI_Var::x_comm );
		MPI_Recv(&local_MM.pElems[0],										1,	MPI_Var::local_x_plane_red,	MPI_Var::prev_x_rank,	MPI_Var::tag,	MPI_Var::x_comm, &MPI_Var::status);
		
		//SEND/RECV BACK ROW//
		MPI_Send(&local_MM.pElems[local_NN.y*local_NN.z],					1,	MPI_Var::local_x_plane_blk,	MPI_Var::prev_x_rank,	MPI_Var::tag,	MPI_Var::x_comm );
		MPI_Recv(&local_MM.pElems[(local_NN.x-1)*local_NN.y*local_NN.z],	1,	MPI_Var::local_x_plane_blk,	MPI_Var::next_x_rank,	MPI_Var::tag,	MPI_Var::x_comm, &MPI_Var::status);
		
		//SEND/RECV LEFT ROW//
		MPI_Send(&local_MM.pElems[(local_NN.y-2)	*local_NN.z],			1,	MPI_Var::local_y_plane_red,	MPI_Var::next_y_rank,	MPI_Var::tag,	MPI_Var::y_comm );
		MPI_Recv(&local_MM.pElems[0],										1,	MPI_Var::local_y_plane_red,	MPI_Var::prev_y_rank,	MPI_Var::tag,	MPI_Var::y_comm, &MPI_Var::status);
		
		//SEND/RECV RIGHT ROW//
		MPI_Send(&local_MM.pElems[local_NN.z],								1,	MPI_Var::local_y_plane_blk,	MPI_Var::prev_y_rank,	MPI_Var::tag,	MPI_Var::y_comm );
		MPI_Recv(&local_MM.pElems[(local_NN.y-1)*local_NN.z],				1,	MPI_Var::local_y_plane_blk,	MPI_Var::next_y_rank,	MPI_Var::tag,	MPI_Var::y_comm, &MPI_Var::status);
		
		//SEND/RECV TOP ROW//
		MPI_Send(&local_MM.pElems[local_NN.z-2],							1,	MPI_Var::local_z_plane_red,	MPI_Var::next_z_rank,	MPI_Var::tag,	MPI_Var::z_comm );
		MPI_Recv(&local_MM.pElems[0],										1,	MPI_Var::local_z_plane_red,	MPI_Var::prev_z_rank,	MPI_Var::tag,	MPI_Var::z_comm, &MPI_Var::status);
		
		//SEND/RECV BOTTOM ROW//
		MPI_Send(&local_MM.pElems[1],										1,	MPI_Var::local_z_plane_blk,	MPI_Var::prev_z_rank,	MPI_Var::tag,	MPI_Var::z_comm );
		MPI_Recv(&local_MM.pElems[local_NN.z-1],							1,	MPI_Var::local_z_plane_blk,	MPI_Var::next_z_rank,	MPI_Var::tag,	MPI_Var::z_comm, &MPI_Var::status);
	}
}; // Update Ghost Rows

void SorSolution::Solve(ResGrid dd, SizeGrid local_N, const CMatrix3D& local_Un, CMatrix3D& local_phi)
{
	double start, finish, runtime(0);
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
	
	double		rres	= 0;
	double		rrb		= cos( M_PI / (2*local_N.max()+2));
//	double		rrb		= cos( M_PI / local_N.max());
	double		wwb		= 2 / ( 1 + sqrt( 1 - pow(rrb,2) ) );
	double		local_ErrNum;															// Numerator of the Remaining Error
	double		ErrNum	= 0;															// Numerator of the Remaining Error
	double		Err		= 0;															// Remaining Error
	int			step	= 0;															// Step Counter
	
	ErrNum	= ErrDen;
	Err		= ErrNum/ErrDen;
	
	MPI_Barrier(MPI_COMM_WORLD);
	start	= MPI_Wtime();
	
	while(Err>epsilon && step<MaxStep)
	{
		local_ErrNum	= 0;
				
		//UPDATE RED POINTS//
		Update_GhostPlanes(local_phi,local_N,red);
		
		//CALCULATE BLACK POINTS//
		for(int kk=1 ; kk<local_N.z-1 ; kk++) for(int jj=1 ; jj<local_N.y-1 ; jj++) for(int ii=1 ; ii<local_N.x-1 ; ii++)
			if((ii+jj+kk)%2==1)
			{
				if (type == ChargeDistribution && local_Un[ii][jj][kk] == 0)
				{
					rres =
					a * local_phi[ii-1][jj][kk]	+ b * local_phi[ii+1][jj][kk] + 
					c * local_phi[ii][jj-1][kk]	+ d * local_phi[ii][jj+1][kk] + 
					e * local_phi[ii][jj][kk-1]	+ f * local_phi[ii][jj][kk+1] + 
					g * local_phi[ii][jj][kk]	- h[ii][jj][kk];
					local_phi[ii][jj][kk] -= wwb*rres;
					local_ErrNum		 += rres*rres;
				}
				else if (type == PotentialDistribution && local_Un[ii][jj][kk] == 0)
				{
					rres = 
					a * local_phi[ii-1][jj][kk]	+ b * local_phi[ii+1][jj][kk] + 
					c * local_phi[ii][jj-1][kk]	+ d * local_phi[ii][jj+1][kk] + 
					e * local_phi[ii][jj][kk-1]	+ f * local_phi[ii][jj][kk+1] + 
					g * local_phi[ii][jj][kk];
					local_phi[ii][jj][kk] -= wwb*rres;
					local_ErrNum += pow(rres,2);
				}
			};	 
		
		//UPDATE BLACK POINTS//
		Update_GhostPlanes(local_phi,local_N,black);

		//CALCULATE RED POINTS//
		for(int kk=1 ; kk<local_N.z-1 ; kk++) for(int jj=1 ; jj<local_N.y-1 ; jj++) for(int ii=1 ; ii<local_N.x-1 ; ii++)
			if((ii+jj+kk)%2==0)
			{
				if (type == ChargeDistribution && local_Un[ii][jj][kk] == 0)
				{
					rres =
					a * local_phi[ii-1][jj][kk]	+ b * local_phi[ii+1][jj][kk] + 
					c * local_phi[ii][jj-1][kk]	+ d * local_phi[ii][jj+1][kk] + 
					e * local_phi[ii][jj][kk-1]	+ f * local_phi[ii][jj][kk+1] + 
					g * local_phi[ii][jj][kk]	- h[ii][jj][kk];
					local_phi[ii][jj][kk]	-= wwb*rres;
					local_ErrNum			+= (rres*rres);
				}
				else if (type == PotentialDistribution && local_Un[ii][jj][kk] == 0)
				{
					rres = 
					a * local_phi[ii-1][jj][kk]	+ b * local_phi[ii+1][jj][kk] + 
					c * local_phi[ii][jj-1][kk]	+ d * local_phi[ii][jj+1][kk] + 
					e * local_phi[ii][jj][kk-1]	+ f * local_phi[ii][jj][kk+1] + 
					g * local_phi[ii][jj][kk];
					local_phi[ii][jj][kk]	-= wwb*rres;
					local_ErrNum			+= pow(rres,2);
				}
			};
		
		//DERIVE GLOBAL ERROR//
		if((step+1)%1 == 0)
			MPI_Allreduce(&local_ErrNum, &ErrNum, 1,	MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		Err = ErrNum/ErrDen;
//		if(MPI_Var::world_rank == MPI_Var::root) printf("step == %3d; ErrNum = %e; ErrDen = %e; Err = %e;\n",step,local_ErrNum,ErrDen,Err);
		step++;
	};
	
	MPI_Barrier(MPI_COMM_WORLD);
	finish	 = MPI_Wtime();
	runtime	+= (finish - start);
	
	if(step==MaxStep)
	{
		if(MPI_Var::world_rank == MPI_Var::root)
		{
			cout<<"*** Allowed computation time exceeded.\n";
			cout<<"*** Maximum allowed iteration step reached.\n";
			cout<<"*** Precision of the result : "<< Err<<".\n";
			cout<<"@ step = "<<step<<" local_ErrNum = "<<local_ErrNum<<" local_ErrDen = "<< ErrDen<<" Err = "<<Err<<endl;
		}
	}
	
	if(MPI_Var::world_rank==MPI_Var::root)	
	{
		printf("Run time for SOR solver: %e s\n",runtime);
		printf("steps in SOR solver    : %d\n",step);
	}
};
/**************************************************************************************/
	