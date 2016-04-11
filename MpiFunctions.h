/*
 *  Utils.h
 *  Created by Jeremy Riousset on 11/19/07.
 */

#ifndef MPIFUNCTIONS_H
#define MPIFUNCTIONS_H
#include "Input.h"
#include "MpiInput.h"

class MPI_foo{
public:
	static	void		CreateComm(void);
	static	void		CreateGridComm(void);
	static	void		CreateCartComm(void);
	static	void		InitLocalDimensions(void);
	static	void		FreeComm(void);
	static	void		CreateLocalPlanes(void);
	
	static	void		Reduce( CMatrix3D&	MM);
	static	CMatrix3D	Scatter(CMatrix3D	MM);
	static	Charge		Scatter(Charge		CC);
	static	Potential	Scatter(Potential	PP);
	static	CMatrix3D	Gather(	CMatrix3D	local_MM);
};

#endif // MPIFUNCTIONS_H
