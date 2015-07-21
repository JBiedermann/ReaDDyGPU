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
package readdy.impl.assembly;

import readdy.api.assembly.IDiffusionEngineFactory;
import readdy.api.sim.core.bd.IDiffusionEngine;
import readdy.api.sim.core.bd.INoiseDisplacementComputer;
import readdy.api.sim.core.bd.IPotentialDisplacementComputer;
import readdy.impl.sim.core.bd.PotentialDisplacementComputer;
import readdy.api.sim.core.particle.IParticleParameters;
import readdy.api.sim.core.pot.IPotentialManager;
import readdy.impl.sim.core.bd.DiffusionEngine;
import readdy.impl.sim.core.bd.NoiseDisplacementComputer;

/**
 *
 * @author schoeneberg
 */
public class DiffusionEngineFactory implements IDiffusionEngineFactory {

    IPotentialManager potentialManager = null;
    IParticleParameters particleParameters = null;

    public void set_potentialManager(IPotentialManager potentialManager) {
        this.potentialManager = potentialManager;
    }

    public void set_particleParameters(IParticleParameters particleParameters) {
        this.particleParameters = particleParameters;
    }

    public IDiffusionEngine createDiffusionEngine() {
        DiffusionEngine diffEngine = new DiffusionEngine();
        diffEngine.set_potentialManager(potentialManager);
        diffEngine.set_particleParameters(particleParameters);

        IPotentialDisplacementComputer pdc = new PotentialDisplacementComputer();
        diffEngine.set_potentialDisplacementComputer(pdc);
        INoiseDisplacementComputer ndc = new NoiseDisplacementComputer();
        diffEngine.set_noiseDisplacementComputer(ndc);

        return diffEngine;
    }
}
