/*
 *  main.cpp
 *  Created by Jeremy Riousset on 11/19/07.
 */

#include "BoundaryConditions.h"

int main(int argc, char** argv)
{
	/*DEFINE SIMULATION VARIABLES**************************************************/
	Var::N.x =  98; // WE MUST HAVE N.x and N.y even (else it unbalance the load on the processes)
	Var::N.y =  98;
	Var::N.z =  98;
	Var::L.init(9.7,9.7,9.7);
	Var::d.init(Var::L,Var::N);
	/******************************************************************************/
	
	/*START MPI********************************************************************/
	MPI_Init(&argc, &argv);
	/******************************************************************************/
	
	//CREATE COMMUNICATORS AND TOPOLOGIES//
	MPI_foo::CreateComm();
	MPI_foo::CreateGridComm();
	MPI_foo::CreateCartComm();
	
	/**********************************************************************************/
	
	MPI_foo::InitLocalDimensions();
	MPI_foo::CreateLocalPlanes();
	MPI_Var::is = (MPI_Var::x_rank != 0);
	MPI_Var::ie = (MPI_Var::x_rank == MPI_Var::dim_sizes[0]-1)*(Var::local_N.x-1) + (MPI_Var::x_rank != MPI_Var::dim_sizes[0]-1)*(Var::local_N.x-2);
	MPI_Var::js = (MPI_Var::y_rank != 0);
	MPI_Var::je = (MPI_Var::y_rank == MPI_Var::dim_sizes[1]-1)*(Var::local_N.y-1) + (MPI_Var::y_rank != MPI_Var::dim_sizes[1]-1)*(Var::local_N.y-2);
	MPI_Var::ks = (MPI_Var::z_rank != 0);
	MPI_Var::ke = (MPI_Var::z_rank == MPI_Var::dim_sizes[2]-1)*(Var::local_N.z-1) + (MPI_Var::z_rank != MPI_Var::dim_sizes[2]-1)*(Var::local_N.z-2);
	
	/**********************************************************************************/
	/*
	if (MPI_Var::world_rank == MPI_Var::root)
	{
		Var::phi.init(Var::N.x,Var::N.y,Var::N.z);
		for(int ii=0 ; ii<Var::N.x ; ii++) for(int jj=0 ; jj<Var::N.y ; jj++) for(int kk=0 ; kk<Var::N.z ; kk++)
			Var::phi[ii][jj][kk] = ii*Var::N.y*Var::N.z + jj*Var::N.z + kk;
	}
	if (MPI_Var::x_rank == 1)
	{
		for(int ii=0 ; ii<Var::N.x ; ii++) for(int jj=0 ; jj<Var::N.y ; jj++) 
		{
			MPI_Var::top_plane[ii][jj] = 200;
			MPI_Var::bot_plane[ii][jj] = 100;
		}
		for(int ii=0 ; ii<Var::N.x ; ii++) for(int kk=0 ; kk<Var::N.z ; kk++) 
		{
			MPI_Var::left_plane[ii][kk] = 300;
			MPI_Var::right_plane[ii][kk] = 400;
		}
		for(int jj=0 ; jj<Var::N.y ; jj++) for(int kk=0 ; kk<Var::N.z ; kk++) 
		{
			MPI_Var::back_plane[jj][kk] = 500;
			MPI_Var::front_plane[jj][kk] = 600;
		}
	}
	
	if (MPI_Var::x_rank == 1)
	{
		MPI_Send(&MPI_Var::top_plane[0][0],		Var::N.x*Var::N.y,	MPI_DOUBLE,	0,	0,	MPI_Var::x_comm );
		MPI_Send(&MPI_Var::bot_plane[0][0],		Var::N.x*Var::N.y,	MPI_DOUBLE,	0,	1,	MPI_Var::x_comm );
		MPI_Send(&MPI_Var::left_plane[0][0],	Var::N.x*Var::N.z,	MPI_DOUBLE,	0,	2,	MPI_Var::x_comm );
		MPI_Send(&MPI_Var::right_plane[0][0],	Var::N.x*Var::N.z,	MPI_DOUBLE,	0,	3,	MPI_Var::x_comm );
		MPI_Send(&MPI_Var::back_plane[0][0],	Var::N.y*Var::N.z,	MPI_DOUBLE,	0,	4,	MPI_Var::x_comm );
		MPI_Send(&MPI_Var::front_plane[0][0],	Var::N.y*Var::N.z,	MPI_DOUBLE,	0,	5,	MPI_Var::x_comm );
	}
	if (MPI_Var::x_rank == 0)
	{
		MPI_Recv(&Var::phi[0][0][Var::N.z-1],	1,	MPI_Var::z_plane, 1,	0,	MPI_Var::x_comm, &MPI_Var::status);		
		MPI_Recv(&Var::phi[0][0][0],			1,	MPI_Var::z_plane, 1,	1,	MPI_Var::x_comm, &MPI_Var::status);		
		MPI_Recv(&Var::phi[0][0][0],			1,	MPI_Var::y_plane, 1,	2,	MPI_Var::x_comm, &MPI_Var::status);		
		MPI_Recv(&Var::phi[0][Var::N.y-1][0],	1,	MPI_Var::y_plane, 1,	3,	MPI_Var::x_comm, &MPI_Var::status);		
		MPI_Recv(&Var::phi[0][0][0],			1,	MPI_Var::x_plane, 1,	4,	MPI_Var::x_comm, &MPI_Var::status);		
		MPI_Recv(&Var::phi[Var::N.x-1][0][0],	1,	MPI_Var::x_plane, 1,	5,	MPI_Var::x_comm, &MPI_Var::status);		
	}
	*/
	
	if (MPI_Var::world_rank == MPI_Var::root)
		cout<<Var::phi;
	
	/**********************************************************************************/	
	
	//CREATE CHARGE DENSITY ON PROCESS ROOT//
	if (MPI_Var::world_rank == MPI_Var::root)
	{
		Var::Un.init( Var::N.x,Var::N.y,Var::N.z);
		Var::phi.init(Var::N.x,Var::N.y,Var::N.z);
		Var::rho.init(Var::N.x,Var::N.y,Var::N.z);
		Var::Q   = 100;
		Var::Xq  = Var::L.x/2;
		Var::Yq  = Var::L.y/2;
		Var::Zq  = Var::L.z/2;
		Var::Rq1 = 5;
		
		Var::C.sphere(Var::Q, Var::Xq,Var::Yq,Var::Zq, Var::Rq1, Var::d,Var::N);
	}
//	MPI_foo::InitLocalDimensions();
	MPI_foo::CreateLocalPlanes();
	Var::z_gnd = 3;
	
	//SCATTER MATRICES//
	Var::local_C    = MPI_foo::Scatter(Var::C);
	Var::local_Un   = MPI_foo::Scatter(Var::Un);
	
	BC::Apply(	0, Var::phi, Var::local_C.rho);
	Var::local_phi  = MPI_foo::Scatter(Var::phi);
	
	//SOLVE POISSON'S EQUATION//
	SorSolution		SOR;
	SOR.init(Var::local_phi, Var::epsilon, Var::MaxStep, Var::d, Var::local_N, Var::local_C, Var::local_Un);
	SOR.Solve(Var::d, Var::local_N, Var::local_Un, Var::local_phi);
	
	//GATHER SCATTERED MATRICES//
	Var::phi =	MPI_foo::Gather(Var::local_phi);
	
	//STORE SOLUTION//
	if (MPI_Var::world_rank == MPI_Var::root)
	{
		CMatrix1D phi1D(Var::N.z);
		for(int kk=0 ; kk<Var::N.z ; kk++)
			phi1D[kk] = Var::phi[Var::N.x/2][Var::N.y/2][kk];
		
		phi1D.fwrite("results/phiNum.dat");
		Var::phi.fwrite("results/phi.dat");
	}
	
	if (MPI_Var::world_rank == MPI_Var::root)
	{
		CMatrix1D Theoretical;
		Theoretical = Var::C.MultipoleAnalyticalSolution(Var::d, Var::N);
		Theoretical.fwrite("results/phiAna.dat");		
	}
	/*STOP MPI*********************************************************************/
	MPI_foo::FreeComm();
	MPI_Finalize();
	/******************************************************************************/
	
	return 0;
}
