// This file is part of V-REP, the Virtual Robot Experimentation Platform.
// 
// Copyright 2006-20 Coppelia Robotics GmbH. All rights reserved.
// marc@coppeliarobotics.com
// www.coppeliarobotics.com
// 
// V-REP is dual-licensed, under the terms of EITHER (at your option):
//   1. V-REP commercial license (contact us for details)
//   2. GNU GPL (see below)
// 
// GNU GPL license:
// -------------------------------------------------------------------
// V-REP is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// V-REP IS DISTRIBUTED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
// WARRANTY. THE USER WILL USE IT AT HIS/HER OWN RISK. THE ORIGINAL
// AUTHORS AND COPPELIA ROBOTICS GMBH WILL NOT BE LIABLE FOR DATA LOSS,
// DAMAGES, LOSS OF PROFITS OR ANY OTHER KIND OF LOSS WHILE USING OR
// MISUSING THIS SOFTWARE.
// 
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with V-REP.  If not, see <http://www.gnu.org/licenses/>.
// -------------------------------------------------------------------
//
// This file was automatically created for V-REP release V3.2.1 on May 3rd 2015

#include <QtCore/QCoreApplication>
#include "v_repLib.h"
#include <vector>
#include <QLibrary>
#include <QFileInfo>
#include <QDir>
#include <iostream>
#include <unistd.h>
#include <stdio.h>
extern "C"
{
    #include "Python.h"
    #include "py/nao.h"
}

#ifdef WIN_VREP
#include <direct.h>
#endif

// Following required to have Lua extension libraries work under Linux. Very strange indeed.
//****************************************************************************
#ifdef LINUX_VREP
extern "C" {
    #include "lua.h"
    #include "lualib.h"
    #include "lauxlib.h"
}
void dummyFunction()
{
    lua_State *L;
    L=luaL_newstate();
}
#endif
//****************************************************************************

std::vector<int> pluginHandles;
std::string sceneOrModelOrUiToLoad;
bool autoStart=false;
int simStopDelay=0;
bool autoQuit=false;

int loadPlugin(const char* theName,const char* theDirAndName)
{
	std::cout << "Plugin '" << theName << "': loading...\n";
	int pluginHandle=simLoadModule(theDirAndName,theName);
	if (pluginHandle==-3)
#ifdef WIN_VREP
		std::cout << "Plugin '" << theName << "': load failed (could not load). The plugin probably couldn't load dependency libraries. Try rebuilding the plugin.\n";
#endif
#ifdef MAC_VREP
		std::cout << "Plugin '" << theName << "': load failed (could not load). The plugin probably couldn't load dependency libraries. Try 'otool -L pluginName.dylib' for more infos, or simply rebuild the plugin.\n";
#endif
#ifdef LINUX_VREP
		std::cout << "Plugin '" << theName << "': load failed (could not load). The plugin probably couldn't load dependency libraries. Try 'ldd pluginName.so' for more infos, or simply rebuild the plugin.\n";
#endif

	if (pluginHandle==-2)
		std::cout << "Plugin '" << theName << "': load failed (missing entry points).\n";
	if (pluginHandle==-1)
		std::cout << "Plugin '" << theName << "': load failed (failed initialization).\n";
	if (pluginHandle>=0)
		std::cout << "Plugin '" << theName << "': load succeeded.\n";
	return(pluginHandle);
}
////////////////////////////////////////////////////////////////////////////
const int STD_RUN = 0;
const int TEST_RUN = 1;
const int LEARN_RUN = 2;
const int PLAYBACK_RUN = 3;
const int FIXED_RUN = 4;
const int PAR_SEARCH_RUN = 5;
const int DDPG_RUN = 6;
const int DDPG_TEST_RUN = 7;

int episodes = 1;

int runType = STD_RUN;

bool endEpisode = false;
bool endRun = false;
bool finalize = false;
int episodeCounter = 1;
float episodeTime = 15;
float lastEpisodeTime = 0;
int relevantSize = 12;
int relevantJoints[12] = {0,1,5,6,10,11,14,15,18,19,22,23};

simChar* nao_start_config;

simInt ground_handle;
simInt NAO_handle;
simInt NAO_shape_tree_size;
simInt NAO_joint_tree_size;
simInt NAO_force_tree_size;
simInt* NAO_shape_tree;
simInt* NAO_joint_tree;
simInt* NAO_force_tree;

void naoInit(){
    Py_Initialize();
    initnao();
    switch(runType){
    case TEST_RUN : episodes = nao_test_init(); break;
    case LEARN_RUN : episodes = nao_learn_init(); break;
    case PLAYBACK_RUN : nao_playback_init(); break;
    case FIXED_RUN : nao_fixed_init(); break;
    case PAR_SEARCH_RUN : episodes = nao_ddpg_init(); break;
    case DDPG_RUN : episodes = nao_ddpg_init(); break;
    case DDPG_TEST_RUN : episodes = nao_ddpg_test_init(); break;
    }
    printf("RUN TYPE = %i\n", runType);
    if(runType){
        ground_handle = simGetObjectHandle("ResizableFloor_5_25_visibleElement");
        NAO_handle = simGetObjectHandle("NAO");
        NAO_shape_tree = simGetObjectsInTree(NAO_handle,sim_object_shape_type,1,&NAO_shape_tree_size);
        NAO_joint_tree = simGetObjectsInTree(NAO_handle,sim_object_joint_type,1,&NAO_joint_tree_size);
        NAO_force_tree = simGetObjectsInTree(NAO_handle,sim_object_forcesensor_type,1,&NAO_force_tree_size);
        for(int i = 0; i < NAO_joint_tree_size; i++){
            simSetJointPosition(NAO_joint_tree[i], 0);
            simSetJointTargetPosition(NAO_joint_tree[i], 0);
            if(i == 2 || i == 3){
                simSetJointPosition(NAO_joint_tree[i], 1.57);
                simSetJointTargetPosition(NAO_joint_tree[i], 1.57);
            }
        }
        simSaveModel(NAO_handle,"singleNAO.ttm");
        nao_start_config = simGetConfigurationTree(NAO_handle);

    }
}

void naoEnd(){
    Py_Finalize();
}

void resetDynamics(int main_handle){
    for(int i=0;;i++){
        int handle = simGetObjectChild(main_handle, i);
        if (handle < 0) break;
        printf("%s\n", simGetObjectName(handle));
        simResetDynamicObject(handle);
        resetDynamics(handle);
    }
}

void naoReset(){
    simRemoveModel(NAO_handle);
    NAO_handle = simLoadModel("singleNAO.ttm");
    NAO_shape_tree = simGetObjectsInTree(NAO_handle,sim_object_shape_type,1,&NAO_shape_tree_size);
    NAO_joint_tree = simGetObjectsInTree(NAO_handle,sim_object_joint_type,1,&NAO_joint_tree_size);
    NAO_force_tree = simGetObjectsInTree(NAO_handle,sim_object_forcesensor_type,1,&NAO_force_tree_size);
    episodeCounter += 1;
    if (episodeCounter > episodes)return;
    endEpisode = false;
//    resetDynamics(NAO_handle);
//    simSetConfigurationTree(nao_start_config);
}

bool naoDetectCollision(){
    for(int i = 0; i < NAO_shape_tree_size; i++){
        if(i == 51 || i == 52)continue;
        if(simGetObjectSpecialProperty(NAO_shape_tree[i])){// if is collidable
//        printf("%s [%d]\n",simGetObjectName(NAO_tree[i]),i);
    simInt collision = simCheckCollision(NAO_shape_tree[i], ground_handle);
    if(collision == 1){
    printf("Collision with %s [%d] at %.2fs\n",simGetObjectName(NAO_shape_tree[i]),i,simGetSimulationTime()-lastEpisodeTime);
    endEpisode = true;
    return true;
    }
}
         }
return false;
}

void naoDetectEndOfEpisode(){
//    printf("time = %.2f, more than %.2f\n",simGetSimulationTime(),lastEpisodeTime+episodeTime);
    if (simGetSimulationTime() > lastEpisodeTime+episodeTime || naoDetectCollision()){
//        printf("end of episode %d\n",episodeCounter);
        printf("episode length: %.2f\n",simGetSimulationTime()-lastEpisodeTime);
        endEpisode = true;
        }
}

void naoDetectEndOfRun(){
    if(episodeCounter > episodes){
        printf("%d %d",episodeCounter,episodes);
        endRun = true;
        return;
    }
    naoDetectEndOfEpisode();
}

float abs(float x){
    if(x > 0){
        return x;
    }
    else{
        return -1*x;
    }
}

float naoCountScore()
{
    simFloat* nao_position = new simFloat[3];
    simGetObjectPosition(NAO_handle,-1,nao_position);
    float x, y, t, score;
    float dt = 5e-2;
    x = nao_position[0];
    y = nao_position[1];
    t = simGetSimulationTime()-lastEpisodeTime;
//    score = x*x + y*y;
    score = x;
//    score = dt;
//    printf("%d. episode score = %.2f (t = %.1f)\n",episodeCounter,score,t);
    return(score);
}
//lastEpisodeTime = simGetSimulationTime();

void naoLearnStep(){
    naoDetectEndOfRun();
    if (endRun){
       if(!finalize){
           finalize = true;
           simStopSimulation();
       }
       return;
    }else if (endEpisode){
       nao_learn_end_episode(naoCountScore());
       lastEpisodeTime = simGetSimulationTime();
       naoReset();
       return;
    }
    std::vector<float> joint_pos(relevantSize*2+3);

    for(int i = 0; i < relevantSize; i++){
        simFloat position, velocity;
        simGetJointPosition(NAO_joint_tree[relevantJoints[i]], &position);
        simGetObjectFloatParameter(NAO_joint_tree[relevantJoints[i]],2012,&velocity);
        joint_pos[i] = (float) position;
        joint_pos[i+relevantSize] = (float) velocity;
         }

    for(int i = 0;i < 1;i++){
    simFloat* forceVector = new simFloat[3];
    simReadForceSensor(NAO_force_tree[i],forceVector,NULL);
    joint_pos[relevantSize*2] = forceVector[0];
    joint_pos[relevantSize*2+1] = forceVector[1];
    joint_pos[relevantSize*2+2] = forceVector[2];
    }
    std::vector<float> joint_target_velocity = nao_learn_step(joint_pos);
    for(int i = 0; i < relevantSize; i++){
            simSetJointTargetPosition(NAO_joint_tree[relevantJoints[i]], joint_target_velocity[i]);
    }
}

void naoTestStep(){
    naoDetectEndOfRun();
    if (endRun){
       if(!finalize){
           finalize = true;
           simStopSimulation();
       }
       return;
    }else if (endEpisode){
       lastEpisodeTime = simGetSimulationTime();
       nao_test_end_episode(naoCountScore());
       naoReset();
       return;
    }
    std::vector<float> joint_pos(relevantSize*2+3);

    for(int i = 0; i < relevantSize; i++){
        simFloat position, velocity;
        simGetJointPosition(NAO_joint_tree[relevantJoints[i]], &position);
        simGetObjectFloatParameter(NAO_joint_tree[relevantJoints[i]],2012,&velocity);
        joint_pos[i] = (float) position;
        joint_pos[i+relevantSize] = (float) velocity;
         }

    for(int i = 0;i < 1;i++){
    simFloat* forceVector = new simFloat[3];
    simReadForceSensor(NAO_force_tree[i],forceVector,NULL);
    joint_pos[relevantSize*2] = forceVector[0];
    joint_pos[relevantSize*2+1] = forceVector[1];
    joint_pos[relevantSize*2+2] = forceVector[2];
    }
    std::vector<float> joint_target_velocity = nao_test_step(joint_pos);
    for(int i = 0; i < relevantSize; i++){
            simSetJointTargetPosition(NAO_joint_tree[relevantJoints[i]], joint_target_velocity[i]);
    }
}

void naoFixedStep(){
    naoDetectEndOfRun();
    if (endRun){
       if(!finalize){
           finalize = true;
           simStopSimulation();
       }
       return;
    }
    std::vector<float> joint_pos(relevantSize*2+3);

    for(int i = 0; i < relevantSize; i++){
        simFloat position, velocity;
        simGetJointPosition(NAO_joint_tree[relevantJoints[i]], &position);
        simGetObjectFloatParameter(NAO_joint_tree[relevantJoints[i]],2012,&velocity);
        joint_pos[i] = (float) position;
        joint_pos[i+relevantSize] = (float) velocity;
         }

    for(int i = 0;i < 1;i++){
    simFloat* forceVector = new simFloat[3];
    simReadForceSensor(NAO_force_tree[i],forceVector,NULL);
    joint_pos[relevantSize*2] = forceVector[0];
    joint_pos[relevantSize*2+1] = forceVector[1];
    joint_pos[relevantSize*2+2] = forceVector[2];
//    printf("%s:x=%.5f,y=%.5f,z=%.5f\n",simGetObjectName(NAO_force_tree[i]),forceVector[0],forceVector[1],forceVector[2]);
    }
    std::vector<float> joint_target_velocity = nao_fixed_step(joint_pos);
    for(int i = 0; i < relevantSize; i++){
            simSetJointTargetPosition(NAO_joint_tree[relevantJoints[i]], joint_target_velocity[i]);
    }
}

void naoPlaybackStep(){
    if (simGetSimulationTime() < 0){
       return;
    }
    if (simGetSimulationTime() > 70){
       nao_playback_finish();
       runType = STD_RUN;
       return;
    }

    std::vector<float> joint_pos(relevantSize*2+3);

    for(int i = 0; i < relevantSize; i++){
        simFloat position, velocity;
        simGetJointPosition(NAO_joint_tree[relevantJoints[i]], &position);
        simGetObjectFloatParameter(NAO_joint_tree[relevantJoints[i]],2012,&velocity);
        joint_pos[i] = (float) position;
        joint_pos[i+relevantSize] = (float) velocity;
         }
    simFloat* forceVector = new simFloat[3];
    simReadForceSensor(NAO_force_tree[0],forceVector,NULL);
    joint_pos[relevantSize*2] = forceVector[0];
    joint_pos[relevantSize*2+1] = forceVector[1];
    joint_pos[relevantSize*2+2] = forceVector[2];

    std::vector<float> joint_target_velocity(relevantSize);
    for(int i = 0; i < relevantSize; i++){
        simFloat velocity;
        simGetObjectFloatParameter(NAO_joint_tree[relevantJoints[i]],2012,&velocity);
//        simGetJointTargetPosition(NAO_joint_tree[relevantJoints[i]],&velocity);
        joint_target_velocity[i] = (float) velocity;
    }
    nao_playback_step(joint_pos,joint_target_velocity);
}

float sgn(float val) {
    return (0 < val) - (val < 0);
}
void naoDDPGStep(){
    naoDetectEndOfRun();
    if (endRun){
       if(!finalize){
           finalize = true;
           simStopSimulation();
       }
       return;
    }else if (endEpisode){
        float t = simGetSimulationTime()-lastEpisodeTime;
        // return time of the episode -> to check if robot fell
        switch(runType){
            case PAR_SEARCH_RUN: nao_par_search_end_episode(t*naoCountScore()); break;
            case DDPG_RUN : nao_ddpg_end_episode(naoCountScore()); break;
            case DDPG_TEST_RUN : nao_ddpg_test_end_episode(naoCountScore()); break;

        };
        lastEpisodeTime = simGetSimulationTime();
        naoReset();
        return;
    }
    std::vector<float> joint_pos(relevantSize*2+3);

    for(int i = 0; i < relevantSize; i++){
        simFloat position, velocity;
        simGetJointPosition(NAO_joint_tree[relevantJoints[i]], &position);
        simGetObjectFloatParameter(NAO_joint_tree[relevantJoints[i]],2012,&velocity);
        joint_pos[i] = (float) position;
        joint_pos[i+relevantSize] = (float) velocity;
         }

    for(int i = 0;i < 1;i++){
    simFloat* forceVector = new simFloat[3];
    simReadForceSensor(NAO_force_tree[i],forceVector,NULL);
    joint_pos[relevantSize*2] = forceVector[0];
    joint_pos[relevantSize*2+1] = forceVector[1];
    joint_pos[relevantSize*2+2] = forceVector[2];
    }
    std::vector<float> joint_target;
    switch(runType){
    case DDPG_RUN : joint_target = nao_ddpg_step(joint_pos, naoCountScore()); break;
    case DDPG_TEST_RUN : joint_target = nao_ddpg_test_step(joint_pos); break;
    case PAR_SEARCH_RUN : joint_target = nao_ddpg_test_step(joint_pos); break;
    }
    for(int i = 0; i < relevantSize; i++){
        if(runType==PAR_SEARCH_RUN){
            simSetJointTargetPosition(NAO_joint_tree[relevantJoints[i]], joint_target[i]*2);
        }else{
//            simSetJointTargetPosition(NAO_joint_tree[relevantJoints[i]], joint_target[i]*2);
            float infty = 10;
            float target_velocity = sgn(joint_target[i]) * infty;
            float target_force = joint_target[i] > 0 ? joint_target[i] : -joint_target[i];
            simSetJointTargetVelocity(NAO_joint_tree[relevantJoints[i]], target_velocity);
            simSetJointForce(NAO_joint_tree[relevantJoints[i]], target_force);
        }
//            simSetJointTargetPosition(NAO_joint_tree[relevantJoints[i]], joint_target[i]);
//            simSetJointTargetVelocity(NAO_joint_tree[relevantJoints[i]], joint_target[i]);
    }
}


void naoStep(){
    switch(runType){
    case TEST_RUN : naoTestStep(); break;
    case LEARN_RUN : naoLearnStep(); break;
    case PLAYBACK_RUN : naoPlaybackStep(); break;
    case FIXED_RUN : naoFixedStep(); break;
    case PAR_SEARCH_RUN : naoDDPGStep(); break;
    case DDPG_RUN : naoDDPGStep(); break;
    case DDPG_TEST_RUN : naoDDPGStep(); break;
    }
}

////////////////////////////////////////////////////////////////////////////
void simulatorInit()
{
	std::cout << "Simulator launched.\n";

	QFileInfo pathInfo(QCoreApplication::applicationFilePath());
	std::string pa=pathInfo.path().toStdString();
	QDir dir(pa.c_str());
	dir.setFilter(QDir::Files|QDir::Hidden); // |QDir::NoSymLinks); // removed on 11/4/2013 thanks to Karl Robillard
	dir.setSorting(QDir::Name);
	QStringList filters;
	int bnl=8;
#ifdef WIN_VREP
	std::string tmp("v_repExt*.dll");
#endif
#ifdef MAC_VREP
	std::string tmp("libv_repExt*.dylib");
	bnl=11;
#endif
#ifdef LINUX_VREP
	std::string tmp("libv_repExt*.so");
	bnl=11;
#endif
	filters << tmp.c_str();
	dir.setNameFilters(filters);
	QFileInfoList list=dir.entryInfoList();
	std::vector<std::string> theNames;
	std::vector<std::string> theDirAndNames;
	for (int i=0;i<list.size();++i)
	{
		QFileInfo fileInfo=list.at(i);
		std::string bla(fileInfo.baseName().toLocal8Bit());
		std::string tmp;
		if (bnl==int(bla.size()))
			tmp="VrepExt"; // This is the extension module of v_rep (exception in naming)!
		else
			tmp.assign(bla.begin()+bnl,bla.end());

		bool underscoreFound=false;
		for (int i=0;i<int(tmp.length());i++)
		{
			if (tmp[i]=='_')
				underscoreFound=true;
		}
		if (!underscoreFound)
		{
			theNames.push_back(tmp);
			theDirAndNames.push_back(fileInfo.absoluteFilePath().toLocal8Bit().data());
		}
	}

	// Load the system plugins first:
	for (int i=0;i<int(theNames.size());i++)
	{
		if ((theNames[i].compare("MeshCalc")==0)||(theNames[i].compare("Dynamics")==0)||(theNames[i].compare("PathPlanning")==0))
		{
			int pluginHandle=loadPlugin(theNames[i].c_str(),theDirAndNames[i].c_str());
			if (pluginHandle>=0)
				pluginHandles.push_back(pluginHandle);
			theDirAndNames[i]=""; // mark as 'already loaded'
		}
	}
	simLoadModule("",""); // indicate that we are done with the system plugins

	// Now load the other plugins too:
	for (int i=0;i<int(theNames.size());i++)
	{
		if (theDirAndNames[i].compare("")!=0)
		{ // not yet loaded
			int pluginHandle=loadPlugin(theNames[i].c_str(),theDirAndNames[i].c_str());
			if (pluginHandle>=0)
				pluginHandles.push_back(pluginHandle);
		}
	}

	if (sceneOrModelOrUiToLoad.length()!=0)
	{ // Here we double-clicked a V-REP file or dragged-and-dropped it onto this application
		int l=int(sceneOrModelOrUiToLoad.length());
		if ((l>4)&&(sceneOrModelOrUiToLoad[l-4]=='.')&&(sceneOrModelOrUiToLoad[l-3]=='t')&&(sceneOrModelOrUiToLoad[l-2]=='t'))
		{
			simSetBooleanParameter(sim_boolparam_scene_and_model_load_messages,1);
			if (sceneOrModelOrUiToLoad[l-1]=='t') // trying to load a scene?
			{
				if (simLoadScene(sceneOrModelOrUiToLoad.c_str())==-1)
					simAddStatusbarMessage("Scene could not be opened.");
			}
			if (sceneOrModelOrUiToLoad[l-1]=='m') // trying to load a model?
			{
				if (simLoadModel(sceneOrModelOrUiToLoad.c_str())==-1)
					simAddStatusbarMessage("Model could not be loaded.");
			}
			if (sceneOrModelOrUiToLoad[l-1]=='b') // trying to load a UI?
			{
				if (simLoadUI(sceneOrModelOrUiToLoad.c_str(),0,NULL)==-1)
					simAddStatusbarMessage("UI could not be loaded.");
			}
			simSetBooleanParameter(sim_boolparam_scene_and_model_load_messages,0);
		}
	}

    naoInit();

}

void simulatorLoop()
{	// The main application loop (excluding the GUI part)
	static bool wasRunning=false;
	int auxValues[4];
	int messageID=0;
	int dataSize;
	if (autoStart)
	{
		simStartSimulation();
		autoStart=false;
	}
	while (messageID!=-1)
	{
		simChar* data=simGetSimulatorMessage(&messageID,auxValues,&dataSize);
		if (messageID!=-1)
		{
			if (messageID==sim_message_simulation_start_resume_request)
				simStartSimulation();
			if (messageID==sim_message_simulation_pause_request)
				simPauseSimulation();
			if (messageID==sim_message_simulation_stop_request)
				simStopSimulation();
			if (data!=NULL)
				simReleaseBuffer(data);
		}
	}

	// Handle a running simulation:
	if ( (simGetSimulationState()&sim_simulation_advancing)!=0 )
	{
		wasRunning=true;
		if ( (simGetRealTimeSimulation()!=1)||(simIsRealTimeSimulationStepNeeded()==1) )
		{
            if ((simHandleMainScript()&sim_script_main_script_not_called)==0){
                simAdvanceSimulationByOneStep();
                naoStep();
//                simAdvanceSimulationByOneStep();
            }
			if ((simStopDelay>0)&&(simGetSimulationTime()>=float(simStopDelay)/1000.0f))
			{
				simStopDelay=0;
				simStopSimulation();
			}
		}
	}
	else
	{
		if (wasRunning&&autoQuit)
		{
			wasRunning=false;
			simQuitSimulator(true); // will post the quit command
		}
	}
}

void simulatorDeinit()
{
    naoEnd();
	// Unload all plugins:
	for (int i=0;i<int(pluginHandles.size());i++)
		simUnloadModule(pluginHandles[i]);
	pluginHandles.clear();
	std::cout << "Simulator ended.\n";
}

int main(int argc, char* argv[])
{
    std::string libLoc(argv[0]);
    while ((libLoc.length()>0)&&(libLoc[libLoc.length()-1]))
    {
        int l=libLoc.length();
        if (libLoc[l-1]!='/')
            libLoc.erase(libLoc.end()-1);
        else
        { // we might have a 2 byte char:
            if (l>1)
            {
                if (libLoc[l-2]>0x7F)
                    libLoc.erase(libLoc.end()-1);
                else
                    break;

            }
            else
                break;
        }
    }
    chdir(libLoc.c_str());

#ifdef WIN_VREP
	LIBRARY lib=loadVrepLibrary("v_rep");
#endif
#ifdef MAC_VREP
	LIBRARY lib=loadVrepLibrary("libv_rep.dylib");
#endif
#ifdef LINUX_VREP
	LIBRARY lib=loadVrepLibrary("libv_rep.so");
#endif

	bool wasRunning=false;
	if (lib!=NULL)
	{
		if (getVrepProcAddresses(lib)!=0)
		{
			int guiItems=sim_gui_all;
			for (int i=1;i<argc;i++)
			{
				std::string arg(argv[i]);
				if (arg[0]=='-')
				{
					if (arg.length()>=2)
					{
						if (arg[1]=='h')
							guiItems=sim_gui_headless;
						if (arg[1]=='s')
						{
							autoStart=true;
							simStopDelay=0;
							unsigned int p=2;
							while (arg.length()>p)
							{
								simStopDelay*=10;
								if ((arg[p]>='0')&&(arg[p]<='9'))
									simStopDelay+=arg[p]-'0';
								else
								{
									simStopDelay=0;
									break;
								}
								p++;
							}
						}
                        /////////////////////////
                        if (arg[1]=='t')
                            runType=TEST_RUN;
                        if (arg[1]=='l')
                            runType=LEARN_RUN;
                        if (arg[1]=='p')
                            runType=PLAYBACK_RUN;
                        if (arg[1]=='f')
                            runType=FIXED_RUN;
                        if (arg=="-par_search")
                            runType=PAR_SEARCH_RUN;
                        if (arg=="-ddpg")
                            runType=DDPG_RUN;
                        if (arg=="-ddpgtest")
                            runType=DDPG_TEST_RUN;
                        /////////////////////////
						if (arg[1]=='q')
							autoQuit=true;
						if ((arg[1]=='a')&&(arg.length()>2))
						{
							std::string tmp;
							tmp.assign(arg.begin()+2,arg.end());
							simSetStringParameter(sim_stringparam_additional_addonscript_firstscene,tmp.c_str()); // normally, never call API functions before simRunSimulator!!
						}
						if ((arg[1]=='b')&&(arg.length()>2))
						{
							std::string tmp;
							tmp.assign(arg.begin()+2,arg.end());
							simSetStringParameter(sim_stringparam_additional_addonscript,tmp.c_str()); // normally, never call API functions before simRunSimulator!!
						}
						if ((arg[1]=='g')&&(arg.length()>2))
						{
							static int cnt=0;
							std::string tmp;
							tmp.assign(arg.begin()+2,arg.end());
							if (cnt<9)
								simSetStringParameter(sim_stringparam_app_arg1+cnt,tmp.c_str()); // normally, never call API functions before simRunSimulator!!
							cnt++;
						}
					}
				}
				else
				{
					if (sceneOrModelOrUiToLoad.length()==0)
						sceneOrModelOrUiToLoad=arg;
				}
			}

			if (simRunSimulator("V-REP",guiItems,simulatorInit,simulatorLoop,simulatorDeinit)!=1)
				std::cout << "Failed initializing and running V-REP\n";
			else
				wasRunning=true;
		}
		else
			std::cout << "Error: could not find all required functions in the V-REP library\n";
		unloadVrepLibrary(lib);
	}
	else
		printf("Error: could not find or correctly load the V-REP library\n");
	if (!wasRunning)
		getchar();
	return(0);
}
