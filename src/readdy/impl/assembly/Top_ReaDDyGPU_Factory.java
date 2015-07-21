
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

import readdy.api.analysis.IAnalysisAndOutputManager;
import readdy.api.assembly.ITopFactory;
import readdy.api.io.in.par_global.IGlobalParameters;
import readdy.api.sim.core.ICore;
import readdy.api.sim.core.particle.IParticleParameters;
import readdy.api.sim.core.pot.IPotentialInventory;
import readdy.api.sim.core.pot.IPotentialManager;
import readdy.api.sim.top.ITop;
import readdy.api.sim.top.group.IGroupConfiguration;
import readdy.api.sim.top.group.IGroupParameters;
import readdy.api.sim.top.rkHandle.IReactionHandler;
import readdy.impl.sim.top.Top;
import readdy.impl.sim.top.TopGPU;

/**
 *
 * @author schoeneberg
 */
public class Top_ReaDDyGPU_Factory implements ITopFactory {
    IParticleParameters particleParameters;
    IPotentialInventory potentialInventory;
    IPotentialManager potentialManager;
    String tplg_grp;
    String tplg_crd;
    String tplgy_potentials;
    String param_particles;
    String param_groups;
    String param_global;
    IGroupParameters groupParameters;
    IGroupConfiguration groupConfiguration;
    

    public void setAnalysisManager(IAnalysisAndOutputManager analysisManager) {
        this.analysisManager = analysisManager;
    }

    public void setCore(ICore core) {
        this.core = core;
    }

    public void setGlobalParameters(IGlobalParameters globalParameters) {
        this.globalParameters = globalParameters;
    }

    public void setParticleParameters(IParticleParameters particleParameters){
        this.particleParameters=particleParameters;
    }
    
    public void setPotentialManager(IPotentialManager potentialManager){
        this.potentialManager=potentialManager;
    }
    
    public void setPotentialInventory(IPotentialInventory potentialInventory){
        this.potentialInventory=potentialInventory;
    }
    
    public void setGroupParameters( IGroupParameters groupParameters){
        this.groupParameters=groupParameters;
    }
    
    public void setGroupConfiguration( IGroupConfiguration groupConfiguration){
        this.groupConfiguration=groupConfiguration;
    }
    
    public void setPathTplgyGrp(String tpgl_grp){
        this.tplg_grp=tpgl_grp;
    }
    
    public void setPathTplgyCrd(String tplg_crd){
        this.tplg_crd=tplg_crd;
    }
    
    public void setPathTplgyPot(String tplgy_potentials){
        this.tplgy_potentials=tplgy_potentials;
    }
    
    public void setPathParamPart(String param_particles){
        this.param_particles=param_particles;
    }
    
    public void setPathParamGrp(String param_groups){
        this.param_groups=param_groups;
    }
    
    public void setPathParamGlob(String param_global){
        this.param_global=param_global;
    }
    
    
    public void setReactionHandler(IReactionHandler reactionHandler) {
        this.reactionHandler = reactionHandler;
    }
    IAnalysisAndOutputManager analysisManager;
    ICore core;
    IGlobalParameters globalParameters;
    IReactionHandler reactionHandler;

   

    public ITop createTop() {
        if (allInputPresent()) {
            TopGPU top = new TopGPU();
            top.set_Core(core);
            top.set_AnalysisManager(analysisManager);
            top.set_GlobalParameters(globalParameters);
            top.set_ReactionHandler(reactionHandler);
            top.setParticleParameters(particleParameters);
            top.setPotentialInventory(potentialInventory);
            top.setPotentialManager(potentialManager);
            top.setGroupParameters(groupParameters);
            top.setGroupConfiguration(groupConfiguration);
            top.setPathTplgyGrp(tplg_grp);
            top.setPathTplgyCrd(tplg_crd);
            top.setPathTplgyPot(tplgy_potentials);
            top.setPathParamPart( param_particles);
            top.setPathParamGrp( param_groups);
            top.setPathParamGlob( param_global);
    

            return top;
        } else {
            throw new RuntimeException("not all input present");
        }
    }

    private boolean allInputPresent() {
        return core != null
                && analysisManager != null
                && globalParameters != null
                && reactionHandler != null;
    }
}
