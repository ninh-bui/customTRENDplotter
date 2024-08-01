#*************************************************************************************
#				TREND Version 5.0
#		   Thermodynamic Reference & Engineering Data
#
#- software for the calculation of thermodynamic and other properties -
#
#Copyright (C) 2020,  Prof. Dr.-Ing. R.Span
#                     Lehrstuhl fuer Thermodynamik
#                     Ruhr-Universitaet Bochum
#                     Universitaetsstr. 150
#                     D-44892 Bochum
#
#Cite as: Span, R.; Beckmüller, R.; Hielscher, S.; Jäger, A.; Mickoleit, E.;
#          Neumann, T.; Pohl S. M.; Semrau, B.; Thol, M. (2020):
#          TREND. Thermodynamic Reference and Engineering Data 5.0.
#          Lehrstuhl für Thermodynamik, Ruhr-Universität Bochum.
#
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see < http://www.gnu.org/licenses/ >.
#
#*************************************************************************************
#
# Written by David Celny, Sven Pohl, Eric Mickoleit & Andreas Jäger
#

from __future__ import print_function
import ctypes as ct
import sys
import platform
import os
import time
import numpy as np

input_length = 12
fluid_length = 30
path_length = 255
unit_length = 20
out_length = 1024

class Fluid:
    def __init__(self,input,calctype,fluids,moles,eos_ind,mix_ind,path,unit,dll_path):
        input = str.encode(input)
        self.input = ct.create_string_buffer(input.ljust(input_length),input_length)
        calctype = str.encode(calctype)
        self.calctype = ct.create_string_buffer(calctype.ljust(input_length),input_length)
        self.nr_fluids = len(fluids)
        #fluid
        fluids_type = (ct.c_char * fluid_length) * fluid_length
        fluids_tmp = fluids_type()

        for i in range(0,fluid_length):
            fluids_tmp[i] = ct.create_string_buffer(b" ".ljust(fluid_length),fluid_length)

        for fid,fluid in enumerate(fluids):
            fluids_tmp[fid] = ct.create_string_buffer(str.encode(fluid.ljust(fluid_length)),fluid_length)
        self.fluid = fluids_tmp
        
        #moles
        moles_type = (fluid_length * ct.c_double)
        moles_tmp = moles_type()
        for mid,mole in enumerate(moles):
            moles_tmp[mid] =  mole
        self.moles_zero = moles_type()
        self.moles = moles_tmp

        #eos_ind
        eos_ind_type = (fluid_length * ct.c_int)
        eos_ind_tmp = eos_ind_type()
        for eid,eos_id in enumerate(eos_ind):
            eos_ind_tmp[eid] =  eos_id
        self.eos_ind = eos_ind_tmp
         
        self.mix_ind = ct.c_int(mix_ind)

        path = str.encode(path)
        path_type = ct.c_char * path_length
        self.path = ct.create_string_buffer(path.ljust(path_length),path_length)
        unit = str.encode(unit)
        unit_type =  ct.c_char * unit_length
        self.unit = ct.create_string_buffer(unit.ljust(unit_length),unit_length)

        # Variables for TREND_SPEC_EOS
        #limits_text_arg
        limits_text_arg_type = (ct.c_char * fluid_length) * fluid_length
        limits_text_arg_tmp = limits_text_arg_type()
        for i in range(0,fluid_length):
            limits_text_arg_tmp[i] = ct.create_string_buffer(b" ".ljust(fluid_length),fluid_length)
        self.limits_text_arg = limits_text_arg_tmp

        #limits_values
        limits_values_type = (ct.c_double*fluid_length) *(fluid_length+1)
        limits_values_tmp = limits_values_type()
        self.limits_values = limits_values_tmp

        #Variables for Flash
        #Phasetype (intent=out)
        phasetype_type = (5 * ct.c_int)
        self.phasetype = phasetype_type()
        #Phasetext (none)
        phasetext_type = (ct.c_char * 4) * 5
        self.phasetext = phasetext_type()
        #x_phase (out)
        x_phase_type = (30 * ct.c_double) * 5
        self.x_phase = x_phase_type()
        #prop_phase (out)
        prop_phase_type = (30 * ct.c_double) * 5
        self.prop_phase = prop_phase_type()
        #prop_overall (out)
        prop_overall_type = (30 * ct.c_double)
        self.prop_overall = prop_overall_type()
        #lnfug_phase (out)
        lnfug_phase_type = (30 * ct.c_double) * 5
        self.lnfug_phase = lnfug_phase_type()
        #chempot_phase (out)
        chempot_phase_type = (30 * ct.c_double) * 5
        self.chempot_phase = chempot_phase_type()
        #phasefrac (out)
        phasefrac_type = (5 * ct.c_double)
        self.phasefrac = (5 * ct.c_double)()
        #PropNameUnit (none)
        prop_name_unit_type = (ct.c_char * 30) * 37 * 3
        self.prop_name_unit = prop_name_unit_type()

        """ Variables for TREND_CALC from Ninh
        
        Defining variables from trend_exports.f90 into Python.
        Library ctype only has the following useable types: c_int, c_double, c_char. For arrays from fortran, it's necessary to define the type of the array and the size of the array like below.
        
        """
        #prop_name (intent=none)
        prop_name_type = (ct.c_char * 30) * 100
        self.prop_name = prop_name_type()

        #prop_list (intent=in), to specify the properties to calculate in integer form
        prop_list_type = (100 * ct.c_int)
        prop_list_tmp = prop_list_type()
        # for pid, prop_id in enumerate(prop_list):
        #     prop_list_tmp[pid] = prop_id
        # self.prop_list = prop_list_tmp

        #results (intent=none)
        results_type = (100 * ct.c_double)
        self.results = results_type()

        # END of Variables for TREND_CALC

        #Variables for PTDIAG
        #t_pts_out
        t_pts_out = (400 * ct.c_double)()
        self.t_pts_out = t_pts_out
        #p_pts_out
        p_pts_out = (400 * ct.c_double)()
        self.p_pts_out = p_pts_out
        #rholiq_pts_out
        rholiq_pts_out = (400 * ct.c_double)()
        self.rholiq_pts_out = rholiq_pts_out
        #rhovap_pts_out
        rhovap_pts_out = (400 * ct.c_double)()
        self.rhovap_pts_out = rhovap_pts_out
        #pointID_pts_out
        pointID_pts_out = (400 * ct.c_int)()
        self.pointID_pts_out = pointID_pts_out     
        #x_pts_out
        x_pts_out = (400 * ct.c_double) * 30
        self.x_pts_out = x_pts_out()   
        
        #Variables for PTXDIAG
        #p_points
        p_pts_arr = (300 * ct.c_double)()
        self.p_points_array = p_pts_arr
        #T_points
        T_pts_arr = (300 * ct.c_double)()
        self.T_points_array = T_pts_arr
        #rhovap_points
        rhovap_pts_arr = (300 * ct.c_double)()
        self.rhovap_points = rhovap_pts_arr
        #rholiq_points
        rholiq_pts_arr = (300 * ct.c_double)()
        self.rholiq_points = rholiq_pts_arr
        #x_points
        x_points = (300 * ct.c_double) * 4
        self.x_points = x_points() 
        
        #Variable for PTX trace
        p_pts_arr = (1001 * ct.c_double)()
        self.p_points_array_trace = p_pts_arr
        #T_points
        T_pts_arr = (1001 * ct.c_double)()
        self.T_points_array_trace = T_pts_arr
        #rhovap_points
        rhovap_pts_arr = (1001 * ct.c_double)()
        self.rhovap_points_trace = rhovap_pts_arr
        #rholiq_points
        rholiq_pts_arr = (1001 * ct.c_double)()
        self.rholiq_points_trace = rholiq_pts_arr
        #x_points
        x_points = (1001 * ct.c_double) * 4
        self.x_points_trace = x_points() 


        if fid==mid and mid==eid:
            pass
        else:
            raise ValueError('NOT SAME LENGTHS OF INPUTS')
        
        handle_ptr_type = ct.c_int
        self.handle_ptr = handle_ptr_type()
        self.dll_path = dll_path
        assert(os.path.exists(self.dll_path))
        self.TrendDLL = ct.windll.LoadLibrary(self.dll_path)

        # set Trend function inputs/outputs
        # init
        self.TrendDLL.TREND_SPEC_EOS_STDCALL.restype = None
        #self.TrendDLL.TREND_SPEC_EOS_STDCALL.argtypes = fluids_type,moles_type,eos_ind_type, ct.c_int, path_type, unit_type, limits_text_arg_type, limits_values_type, ct.c_int, ct.POINTER(handle_ptr_type),ct.c_int,ct.c_int,ct.c_int,ct.c_int
        # Flash
        self.TrendDLL.FLASH3_STDCALL.restype = ct.c_int
        #self.TrendDLL.FLASH3_STDCALL.argtypes = ct.c_char_p, ct.c_double     ,ct.c_double,        fluids_type,moles_type, eos_ind_type,     ct.c_int            ,path_type, unit_type ,phasetype_type  ,phasetext_type,x_phase_type,prop_phase_type ,prop_overall_type ,lnfug_phase_type ,chempot_phase_type ,phasefrac_type ,prop_name_unit_type , ct.c_int,  ct.POINTER(ct.c_int) ,   ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int
                                                #self.input, ct.c_double(pr1), ct.c_double(pr2), self.fluid, self.moles, self.eos_ind, ct.byref(self.mix_ind), self.path, self.unit, self.phasetype, self.phasetext, self.x_phase, self.prop_phase, self.prop_overall, self.lnfug_phase, self.chempot_phase, self.phasefrac, self.prop_name_unit, errorflag, ct.byref(self.handle_ptr), 12, 30, 255, 20, 4, 30
        
        # get the fluid infos via trend_spec_eos
        self.errorflag = ct.c_int(0)
        
        #print(self.fluid,self.moles, self.eos_ind, self.mix_ind,  self.path, self.unit, self.limits_text_arg, self.limits_values,self.errorflag,ct.byref(self.handle_ptr),ct.c_int(30),ct.c_int(255),ct.c_int(20),ct.c_int(30))
        #self.TrendDLL.TREND_SPEC_EOS_STDCALL(self.fluid,self.moles, self.eos_ind, self.mix_ind,  self.path, self.unit, self.limits_text_arg, self.limits_values,self.errorflag,ct.byref(self.handle_ptr),ct.c_int(30),ct.c_int(255),ct.c_int(20),ct.c_int(30))
        self.TrendDLL.TREND_SPEC_EOS_STDCALL(self.fluid,self.moles, self.eos_ind, ct.byref(self.mix_ind),  self.path, self.unit, self.limits_text_arg, ct.byref(self.limits_values),ct.byref(self.errorflag), ct.byref(self.handle_ptr),30,255,20,30)
        self.lim_val = np.array(self.limits_values)
        self.lim = []
        lim_split = np.array(self.limits_text_arg)
        for i in range(lim_split.shape[0]):
            self.lim.append("".join(lim_split[i].astype(str)).strip())

    # setter functions
    def set_input(self,input_in):
        input = str.encode(input_in)
        self.input = ct.create_string_buffer(input.ljust(input_length),input_length)

    def set_calctype(self,calctype_in):
        calctype = str.encode(calctype_in)
        self.calctype = ct.create_string_buffer(calctype.ljust(input_length),input_length)
    
    def set_moles(self,moles_in): # as list or 1d array
        moles_type = (fluid_length * ct.c_double)
        moles_tmp = moles_type()
        for mid,mole in enumerate(moles_in):
            moles_tmp[mid] =  mole
        self.moles = moles_tmp

    def get_calctype(self):
        return str(ct.c_char_p(ct.addressof(self.calctype)).value.decode("utf-8"))


    def get_input(self):
        return str(ct.c_char_p(ct.addressof(self.input)).value.decode("utf-8"))

    def get_unit(self):
        return str(ct.c_char_p(ct.addressof(self.unit)).value.decode("utf-8")).strip()

    # main trend functions
    def TREND_EOS(self,pr1,pr2):
        errorflag = ct.c_int(0)
        self.TrendDLL.TREND_EOS_STDCALL.restype = ct.c_double # !beware required line otherwise you get unsensible output
        return self.TrendDLL.TREND_EOS_STDCALL(self.calctype,
                                               self.input,
                                               ct.byref(ct.c_double(pr1)),
                                               ct.byref(ct.c_double(pr2)),
                                               self.fluid,
                                               self.moles,
                                               self.eos_ind,
                                               ct.byref(self.mix_ind),
                                               self.path,
                                               self.unit,
                                               ct.byref(errorflag),
                                               ct.byref(self.handle_ptr),
                                               12, 12, 30, 255, 20),errorflag

    def TREND_CALC(self,T,D,nrsubst,prop_list):
        errorflag = ct.c_int(0)

        # Ninh: array input requires this (?):
        prop_list_type = (100 * ct.c_int)
        prop_list_tmp = prop_list_type()
        for pid, prop_id in enumerate(prop_list):
            prop_list_tmp[pid] = prop_id
        self.prop_list = prop_list_tmp


        y = self.TrendDLL.TREND_CALC_STDCALL(ct.byref(ct.c_double(T)),
                                             ct.byref(ct.c_double(D)),
                                             ct.byref(ct.c_int(nrsubst)),
                                             ct.byref(self.prop_list),
                                             self.results,
                                             self.prop_name,
                                             ct.byref(errorflag),
                                             ct.byref(self.handle_ptr),
                                             30)
        """ Ninh (trend_exports.f90)
        Arguments
        ---------
        T: double (temperature)
        D: double (density)
        nrsubst: integer
        prop_list: array of integer(s)
        return: array from self.results, which correponds to the properties in prop_list
        
        Note
        ----
        if nrsubst = 0, function calculate properties in mixture mode
        if nrsubst = 1, function calculate properties of specified component 1 from fluid class in pure mode
        so on...

        For prop_list, refer to TREND manual to get the list of properties and their corresponding number
    
        """
        prop_name=[]

        for p in self.prop_name:
             prop_name.append(p.value.decode("utf-8").strip())

        return np.array(self.results), prop_name, errorflag
    
    def ANC_EQ(self,pr1,pr2):
        errorflag = ct.c_int(0)
        self.TrendDLL.ANC_EQ_STDCALL.restype = ct.c_double # !beware required line otherwise you get unsensible output
        return self.TrendDLL.ANC_EQ_STDCALL(self.calctype, self.input, ct.byref(ct.c_double(pr1)), ct.byref(ct.c_double(pr2)), self.fluid, self.moles, self.eos_ind, ct.byref(self.mix_ind), self.path, self.unit, ct.byref(errorflag), ct.byref(self.handle_ptr), 12, 12, 30, 255, 20)
    
    def FLASH(self,pr1,pr2):
        errorflag = ct.c_int(0)
        #y = self.TrendDLL.FLASH3_STDCALL(self.input, pr1,pr2, self.fluid, self.moles, self.eos_ind, self.mix_ind, self.path, self.unit, self.phasetype, self.phasetext, self.x_phase, self.prop_phase, self.prop_overall, self.lnfug_phase, self.chempot_phase, self.phasefrac, self.prop_name_unit, errorflag, ct.byref(self.handle_ptr), 12, 30, 255, 20, 4, 30)       
        y = self.TrendDLL.FLASH3_STDCALL(self.input,
                                         ct.byref(ct.c_double(pr1)),
                                         ct.byref(ct.c_double(pr2)),
                                         self.fluid, self.moles, self.eos_ind, ct.byref(self.mix_ind), self.path, self.unit, self.phasetype, self.phasetext, self.x_phase, self.prop_phase, self.prop_overall, self.lnfug_phase, self.chempot_phase, self.phasefrac, self.prop_name_unit,
                                         ct.byref(errorflag),
                                         ct.byref(self.handle_ptr),
                                         12, 30, 255, 20, 4, 30)
        #return self.phasetype, self.prop_phase, self.prop_overall
        
        phasetext = []
        prop_name_unit = []
        for p in self.phasetext:
            phasetext.append(p.value.decode("utf-8").strip())
        for p in self.prop_name_unit[1]:
            prop_name_unit.append(p.value.decode("utf-8").strip())

        result = {'phasetype': np.array(self.phasetype), 
                  'phasetext': phasetext,
                  'prop_phase': np.array(self.prop_phase),
                  'prop_overall': np.array(self.prop_overall),
                  'lnfug_phase': np.array(self.lnfug_phase),
                  'chempot_phase': np.array(self.chempot_phase),
                  'phasefrac': np.array(self.phasefrac),
                  'x_phase': np.array(self.x_phase),
                  'prop_name_unit': prop_name_unit}
        return result
 
    def PTDIAG(self,env_pv,p_spec,T_spec,fileout):
        errorflag = ct.c_int(0)
        fileout = str.encode(fileout)
        fileout_arg = ct.create_string_buffer(fileout.ljust(255),255)
        y = self.TrendDLL.PTDIAG_OUT_STDCALL(ct.byref(ct.c_int(env_pv)), self.fluid, self.moles, self.eos_ind, ct.byref(self.mix_ind), self.path, ct.byref(ct.c_double(p_spec)), ct.byref(ct.c_double(T_spec)),self.t_pts_out, self.p_pts_out, self.rholiq_pts_out, self.rhovap_pts_out, self.pointID_pts_out, self.x_pts_out, fileout_arg, ct.byref(errorflag), ct.byref(self.handle_ptr), 30, 255, 255)

        # 30.07.24 Ninh: change position of rhovap and rho liq, so it's the same as ptxdiag
        # also, changed order p_pts_out with t_pts_out.
        data = np.array([self.p_pts_out, self.t_pts_out, self.rhovap_pts_out, self.rholiq_pts_out])

        # Ninh: added self.x_pts_out
        return  data, np.array(self.pointID_pts_out),errorflag, np.array(self.x_pts_out)
          
    def PTXDIAG(self,pr1,fileout):
        #errorflag = ct.c_int(0)
        points = ct.c_int(0)
        fileout = str.encode(fileout)
        fileout_arg = ct.create_string_buffer(fileout.ljust(255),255)
        y = self.TrendDLL.PTXDIAG_OUT_STDCALL(self.input, ct.byref(ct.c_double(pr1)), self.fluid, self.eos_ind, ct.byref(self.mix_ind), self.path, self.p_points_array, self.T_points_array, self.x_points, self.rhovap_points, self.rholiq_points, ct.byref(points), fileout_arg, ct.byref(self.errorflag), ct.byref(self.handle_ptr), 12, 30, 255, 255) 
        data = np.array( [self.p_points_array, self.T_points_array, self.rhovap_points, self.rholiq_points])

        return data, np.array(self.x_points), np.int64(points),self.errorflag
    
    def PTXDIAG_trace(self,pr1,fileout):
        #errorflag = ct.c_int(0)
        points = ct.c_int(0)
        trace = ct.c_int(1)
        fileout = str.encode(fileout)
        fileout_arg = ct.create_string_buffer(fileout.ljust(255),255)
        y = self.TrendDLL.PTXDIAG_ISOCH_OUT(self.input, ct.byref(ct.c_double(pr1)),trace, self.fluid, self.eos_ind, ct.byref(self.mix_ind), self.path, self.p_points_array_trace, self.T_points_array_trace, self.x_points_trace, self.rhovap_points_trace, self.rholiq_points_trace, ct.byref(points), fileout_arg, ct.byref(self.errorflag), ct.byref(self.handle_ptr), 12, 30, 255, 255) 
        data = np.array( [self.p_points_array_trace, self.T_points_array_trace, self.rhovap_points_trace, self.rholiq_points_trace])
        return data, np.array(self.x_points_trace), np.int64(points),self.errorflag
    


    def ALLPROP(self,input_in,pr1,pr2):
        input = str.encode(input_in)
        input = ct.create_string_buffer(input.ljust(input_length),input_length)
        errorflag = ct.c_int(0)
        TEOS = ct.c_double(0)
        DEOS = ct.c_double(0)
        PEOS = ct.c_double(0)
        HEOS = ct.c_double(0)
        SEOS = ct.c_double(0)
        CVEOS = ct.c_double(0)
        UEOS = ct.c_double(0)
        CPEOS = ct.c_double(0)
        CP0EOS = ct.c_double(0)
        QEOS = ct.c_double(0)
        BEOS = ct.c_double(0)
        CEOS = ct.c_double(0)
        WSEOS = ct.c_double(0)
        GEOS = ct.c_double(0)
        AEOS = ct.c_double(0)
        y = self.TrendDLL.ALLPROP_STDCALL(input, ct.byref(ct.c_double(pr1)), ct.byref(ct.c_double(pr2)), self.fluid, self.moles, self.eos_ind, ct.byref(self.mix_ind), self.path, self.unit, \
            ct.byref(TEOS), \
             ct.byref(DEOS), \
             ct.byref(PEOS), \
             ct.byref(UEOS),
			 ct.byref(HEOS),  \
			 ct.byref(SEOS),  \
			 ct.byref(GEOS),  \
			 ct.byref(AEOS),  \
			 ct.byref(CPEOS), \
			 ct.byref(CVEOS), \
			 ct.byref(WSEOS), \
			 ct.byref(BEOS),  \
			 ct.byref(CEOS),  \
			 ct.byref(CP0EOS),\
			 ct.byref(QEOS), \
             ct.byref(errorflag), ct.byref(self.handle_ptr),  12, 30, 255, 20)
        return {"T": TEOS.value ,"D": DEOS.value ,"P": PEOS.value,"U": UEOS.value,"H": HEOS.value,"S":SEOS.value, \
            "G":GEOS.value,"A":AEOS.value,"CP":CPEOS.value,"CV":CVEOS.value,"WS":WSEOS.value,"B":BEOS.value,"C":CEOS.value,"CP0":CP0EOS.value,"QEOS":QEOS.value}
    
    def CRIT_LINE(self,outfile_in = ''):
        
        outfile_in = str.encode(outfile_in)
        outfile = ct.create_string_buffer(outfile_in.ljust(out_length),out_length)
        # optinal arrays with results have to be added
        y = self.TrendDLL.CRIT_LINE_STDCALL(self.fluid, self.eos_ind, ct.byref(self.mix_ind), self.unit, self.path, outfile)
        
        return np.zeros((1,100))
        
    def rhomixcalc(self,t,p,iphase,nrsubst): 
        rho_est_given = 0.0
        self.TrendDLL.RHOMIXCALC_STDCALL.restype = ct.c_double
        y = self.TrendDLL.RHOMIXCALC_STDCALL(ct.byref(ct.c_double(t)),ct.byref(ct.c_double(p)),ct.byref(ct.c_double(rho_est_given)),ct.byref(ct.c_int(iphase)),ct.byref(ct.c_int(nrsubst)),ct.byref(self.handle_ptr))
        return np.float(y)
    
    def comp_conversion(self):
        #fluids_in, moles_in, EOS_indicator, MIX_indicator, path, unitdefinition, composition, errorflag, gl_handle
        self.TrendDLL.COMP_STDCALL.restype = None
        comp_out = self.moles_zero
        errorflag = ct.c_int(0)
        y = self.TrendDLL.COMP_STDCALL(self.fluid, self.moles, self.eos_ind, ct.byref(self.mix_ind), self.path, self.unit, comp_out ,ct.byref(errorflag), ct.byref(self.handle_ptr), 30, 255, 20)
        return np.array(comp_out)
    
    # deallocate handle on memory
    def destroy_handle(self):
        try:
            y = self.TrendDLL.DESTROY_FLUID_handle_STDCALL(ct.byref(self.handle_ptr))
        except:
            print('Handle could not be resetted')
        return
    
    def get_axy(self,t,d,dtau,ddelta,nrsubst):
        self.TrendDLL.A_XY_STDCALL.restype = ct.c_double
        y = self.TrendDLL.A_XY_STDCALL(ct.byref(ct.c_double(t)),ct.byref(ct.c_double(d)),ct.byref(ct.c_int(ddelta)),ct.byref(ct.c_int(dtau)),ct.byref(ct.c_int(nrsubst)),ct.byref(self.handle_ptr))
        return y
    
    def get_mw(self,nr_compo):
        return self.lim_val[nr_compo,0]

    def get_ttp(self,nr_compo):
        return self.lim_val[nr_compo,1]

    def get_ptp(self,nr_compo):
        return self.lim_val[nr_compo,2]
    
    def get_tc(self,nr_compo):
        return self.lim_val[nr_compo,3]
    
    def get_pc(self,nr_compo):
        return self.lim_val[nr_compo,4]
    
    def get_dc(self,nr_compo):
        return self.lim_val[nr_compo,5]
    
    def get_tmax(self,nr_compo):
        return self.lim_val[nr_compo,8]
    
    def get_rhomax(self,nr_compo):
        return self.lim_val[nr_compo,10]
    
    def get_pmax(self,nr_compo):
        return self.lim_val[nr_compo,9]
    
    def get_fluids(self):
        return [str(self.fluid[i].value.strip(), 'UTF-8') for i in range(self.nr_fluids)]
    
    def get_moles(self):
        return [self.moles[i] for i in range(self.nr_fluids)]
    
    def get_acentric_factor(self,nr_compo):
        return self.lim_val[nr_compo,6]
    
    def get_eos_ind(self):
        return [self.eos_ind[i] for i in range(self.nr_fluids)]