/*
 *  MPI_Input.cpp
 *  Created by Jeremy Riousset on 11/19/07.
 */

#include "MpiInput.h"

/*MPI VARIABLES DECLARATION********************************************************/

/*CREATE COMM*/
int						MPI_Var::root;												// Root process rank
int						MPI_Var::world_rank;										// Rank in WORLD_COMM
int						MPI_Var::n_processes;										// Number of processes

/*CREATE GRID COMMUNICATOR*/
int						MPI_Var::grid_rank[3];										// Rank in grid_communicator
int						MPI_Var::n_dims			= 3;								// 3-D scattering
int						MPI_Var::max_dims		= 3;								// Number of dimensions in the communicator
int						MPI_Var::dim_sizes[3];										// Number of processes in the communicator
int						MPI_Var::coordinates[3];									// Coordinates of the process
int						MPI_Var::wrap_around[3];									// Periodicity of the domain
int						MPI_Var::reorder		= 1;								// Let hardware optimize the choice of processes
MPI_Comm				MPI_Var::grid_comm;											// Grid Communicator

/*CREATE CARTESIAN COMMUNICATOR*/
int						MPI_Var::free_coords[3];									// Allow column index to vary b/w processes of the column communicator
int						MPI_Var::direction		= 0;								// Direction (col or row) to find the neighbors (in 1-D this can only be 0)
int						MPI_Var::displacement	= 1;								// Look for neighbors in +axis (target is above current process, source is below)
int						MPI_Var::prev_x_rank;										// Rank of the source  in x_comm
int						MPI_Var::x_rank;											// Rank of the process in x_comm
int						MPI_Var::next_x_rank;										// Rank of the target  in x_comm
int						MPI_Var::prev_y_rank;										// Rank of the source  in y_comm
int						MPI_Var::y_rank;											// Rank of the process in y_comm
int						MPI_Var::next_y_rank;										// Rank of the target  in y_comm
int						MPI_Var::prev_z_rank;										// Rank of the source  in z_comm
int						MPI_Var::z_rank;											// Rank of the process in z_comm
int						MPI_Var::next_z_rank;										// Rank of the target  in z_comm
MPI_Comm				MPI_Var::x_comm;											// Communicator through x
MPI_Comm				MPI_Var::y_comm;											// Communicator through y
MPI_Comm				MPI_Var::z_comm;											// Communicator through z
int						MPI_Var::is,MPI_Var::ie,MPI_Var::js,MPI_Var::je,MPI_Var::ks,MPI_Var::ke;
																					// local x-, y- and z-limits of the domain
/*CREATE LOCAL PLANES*/
MPI_Datatype			MPI_Var::local_x_plane_red;									// Red points	in the local yz-plane (x-plane)
MPI_Datatype			MPI_Var::local_x_plane_blk;									// Black points in the local yz-plane (x-plane)
MPI_Datatype			MPI_Var::local_y_plane_red;									// Red points	in the local xz-plane (y-plane)
MPI_Datatype			MPI_Var::local_y_plane_blk;									// Black points in the local xz-plane (y-plane)
MPI_Datatype			MPI_Var::local_z_plane_red;									// Red points	in the local xy-plane (z-plane)
MPI_Datatype			MPI_Var::local_z_plane_blk;									// Black points in the local xy-plane (z-plane)

MPI_Datatype			MPI_Var::x_plane;											// Matrix used for data transmission through ghost rows
MPI_Datatype			MPI_Var::y_plane;											// Matrix used for data transmission through ghost rows
MPI_Datatype			MPI_Var::z_plane;											// Matrix used for data transmission through ghost rows

CMatrix2D				MPI_Var::top_plane;											// Matrix used for data transmission through ghost rows
CMatrix2D				MPI_Var::bot_plane;											// Matrix used for data transmission through ghost rows
CMatrix2D				MPI_Var::front_plane;										// Matrix used for data transmission through ghost rows
CMatrix2D				MPI_Var::back_plane;										// Matrix used for data transmission through ghost rows
CMatrix2D				MPI_Var::left_plane;										// Matrix used for data transmission through ghost rows
CMatrix2D				MPI_Var::right_plane;										// Matrix used for data transmission through ghost rows

/*SEND/RECV PARAMETERS*/
int						MPI_Var::tag;												// Tag for point to point communication
MPI_Status				MPI_Var::status;											// Status of point to point communication
MPI_Request				MPI_Var::request;											// Request for non-blocking point to point communication
/**********************************************************************************/
