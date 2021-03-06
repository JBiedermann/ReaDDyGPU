/*===========================================================================*\
*           ReaDDy - The Library for Reaction Diffusion Dynamics              *
* =========================================================================== *
* Copyright (c) 2010-2013, Johannes Schöneberg, Frank Noé, FU Berlin          *
* All rights reserved.                                                        *
*                                                                             *
* Redistribution and use in source and binary forms, with or without          *
* modification, are permitted provided that the following conditions are met: *
*                                                                             *
*     * Redistributions of source code must retain the above copyright        *
*       notice, this list of conditions and the following disclaimer.         *
*     * Redistributions in binary form must reproduce the above copyright     *
*       notice, this list of conditions and the following disclaimer in the   *
*       documentation and/or other materials provided with the distribution.  *
*     * Neither the name of Johannes Schöneberg or Frank Noé or the FU Berlin *
*       nor the names of its contributors may be used to endorse or promote   *
*       products derived from this software without specific prior written    *
*       permission.                                                           *
*                                                                             *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" *
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE   *
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE  *
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE   *
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR         *
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF        *
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS    *
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN     *
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)     *
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE  *
* POSSIBILITY OF SUCH DAMAGE.                                                 *
*                                                                             *
\*===========================================================================*/
package readdy.api.sim.core.pot;

import readdy.api.sim.core.pot.potentials.IPotential1;
import readdy.api.sim.core.pot.potentials.IPotential2;
import java.util.Iterator;
import readdy.api.sim.core.particle.IParticle;

/**
 *
 * @author schoeneberg
 */
public interface IPotentialManager
{
   
    public void addPotentialByType(int particleTypeId, int potId);
    public void addPotentialByID(int particleId, int potId);
    public void addPotentialByType(int particleTypeId1, int particleTypeId2, int potId);
    public void addPotentialByID(int particleId1, int particleId2, int potId);

    public boolean removePotentialByType(int particleTypeId, int potId);
    public boolean removePotentialByID(int particleId, int potId);
    public boolean removePotentialByType(int particleTypeId1, int particleTypeId2, int potId);
    public boolean removePotentialByID(int particleId1, int particleId2, int potId);

    /**
     * Returns unimolecular potentials with particle p
     * @param p
     * @return
     */
    public Iterator<IPotential1> getPotentials(IParticle p);
    public Iterator<IPotential1> getPotentials(int pTypeId1);

    /**
     * Returns bimolecular potentials between p1 and p2
     * @return
     */
    public Iterator<IPotential2> getPotentials(IParticle p1, IParticle p2);
    public Iterator<IPotential2> getPotentials(int pTypeId1, int pTypeId2);




}
