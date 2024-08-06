import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from src.customfluid import Fluid


class PTX:
	def __init__(self, input_type: str, set_const: float, mixture: list, eos_ind,
	             mixrule: int, path: str,unit: str, dll_path: str, comp_to_plot=1):
		self.input_type = input_type
		self.set_const = set_const
		self.mixture = mixture
		self.eos_ind = eos_ind
		self.mixrule = mixrule
		self.path = path
		self.unit = unit
		self.dll_path = dll_path
		# Set to 2 if you want to plot wrt to second component.
		self.comp_to_plot = comp_to_plot

	def gen_df(self) -> pd.DataFrame:
		""" Clean up data returned from FORTRAN and return a dataframe."""

		# PTX_DIAG does not care actual composition values,
		# for initialization only, does not affect results.
		comp1 = 0.5
		comp2 = 1 - comp1
		molanteile = [comp1, comp2]

		# Initialize the fluid, calctype does not matter for PTX_DIAG.
		fluid_ptx = Fluid(self.input_type, "p", self.mixture, molanteile, self.eos_ind,
		                  self.mixrule, self.path, self.unit, self.dll_path)

		# Execute PTX-DIAG
		# set_const is the constant pressure or temperature
		ptx_pt, ptx_x, ptx_pts, ptx_errorflag = fluid_ptx.PTXDIAG(self.set_const, fileout="")

		# ptx_x is the composition table
		# in this order (according to FORTRAN code):
		# - x[0] vapor of component 1
		# - x[1] vapor of component 2
		# - x[2] liquid of component 1
		# - x[3] liquid of component 2

		# ptx_pt[0] means all elements of pressure row
		idx_of_zero = np.where(ptx_pt[0] == 0)
		# Removing 0 values from ptx_pt and ptx_x
		ptx_pt = np.delete(ptx_pt, idx_of_zero, axis=1)
		# Same operation done on ptx_x,
		# don't want to delete meaningful zeroes,
		# that's why done this way.
		ptx_x = np.delete(ptx_x, idx_of_zero, axis=1)

		if self.comp_to_plot == 1:
			x_vap = ptx_x[0]
			x_liq = ptx_x[2]
		else:
			# Composition wrt to second component.
			x_vap = ptx_x[1]
			x_liq = ptx_x[3]

		rholiq = ptx_pt[2]
		rhovap = ptx_pt[3]

		if self.input_type in ["pliq", "pvap"]:
			# Create dataframe for ptx diagram
			# "column_name": array_name
			ptx_df = pd.DataFrame({
				"t_df": ptx_pt[1],
				"saturated vapor": x_vap,
				"saturated liquid": x_liq
			})

			ptxm_df = ptx_df.melt('t_df', var_name='Phase', value_name='Composition')

			# Put density values in one continuous array.
			# Each sat point has 2 density values
			# for liq and vapor phase.
			rho_df = np.concatenate((rholiq, rhovap))

			# Add density values to dataframe.
			# Temperature values in this array repeat themselves
			# after certain number of rows due to the melt function.
			ptxm_df['rho_df'] = rho_df

		elif self.input_type in ["tliq", "tvap"]:
			ptx_df = pd.DataFrame({
				"p_df": ptx_pt[0],
				"saturated vapor": x_vap,
				"saturated liquid": x_liq
			})

			# TODO: is the melting necessary here
			ptxm_df = ptx_df.melt('p_df', var_name='Phase', value_name='Composition')

			rho_df = np.concatenate((rholiq, rhovap))

			ptxm_df['rho_df'] = rho_df

		return ptxm_df

	def get_df(self) -> pd.DataFrame:
		return self.gen_df()
	# return edited dataframe to main script

	def calc_prop_x_df(self, prop: int) -> pd.DataFrame:
		""" Calculate prop-x and return dataframe.

		Args:
			prop (int): property code for TREND_CALC. Refer to "Prop list" in TREND manual for property codes.

		Returns:
			object: a dataframe with prop_df column added

		"""

		# Since TREND_CALC is used, only accept TD input -> only work with "tvap" or "tliq".
		if self.input_type in ["tliq", "tvap"]:
			raise ValueError("Only temperature-density input is supported for property calculation.")

		ptxm_df = self.gen_df()
		output = []

		# molanteile affects results!
		for t, d, x in zip(ptxm_df["t_df"], ptxm_df["rho_df"], ptxm_df["Composition"]):
			comp1 = x
			comp2 = 1 - comp1
			molanteile = [comp1, comp2]

			# Calctype does not matter here. "cvr" is placeholder.
			fluid = Fluid("td", "cvr", self.mixture, molanteile, self.eos_ind, self.mixrule, self.path, self.unit, self.dll_path)

			res_tmp = fluid.TREND_CALC(t, d, 0, [prop])
			output.append(res_tmp[0][0])

		ptxm_df["prop_df"] = output

		# Check for error values and print them out
		# if (ptxm_df.prop_df < -1000).any().any():
		# 	print("\033[0;91m Attention: TREND error code(s) in ptxm_df ->  removed from dataframe: \033[0m")
		# 	print(ptxm_df[ptxm_df.prop_df < -1000])
		# 	# remove error values from df_calctypeFromSat, to keep plot looking normal, avoid large jumps in range
		# 	ptxm_df = ptxm_df.drop(ptxm_df[ptxm_df.prop_df < -1000].index)

		return ptxm_df

	def ptxplot(self, show_plot: object = False) -> None:
		""" Plot T-x or P-x diagram

		Args:
		 show_plot: if True, show plot in window.

		Returns:
		 object: None. Save plot as png file to directory.
		"""

		ptxm_df = self.gen_ptx_df()

		if self.comp_to_plot == 1:
			x_label = r"$x_{%s}$" % self.mixture[0]
		elif self.comp_to_plot == 2:
			x_label = r"$x_{%s}$" % self.mixture[1]

		sns.set_style("whitegrid")
		sns.set_palette("bright")

		# Set color for saturated lines (based on sns palette).
		palette = {"saturated vapor": "C3", "saturated liquid": "C0"}

		if self.input_type == "pliq" or input == "pvap":
		    title = str(self.set_const) + " MPa | " + str(self.mixture[0]) + "/" + str(self.mixture[1])
		    g = sns.lineplot(y='t_df', x='Composition', hue='Phase', data=ptxm_df, palette=palette)
		    g.set_title("T-x at " + title)

		elif self.input_type == "tliq" or input == "tvap":
		    title = str(self.set_const) + " K | " + str(self.mixture[0]) + "/" + str(self.mixture[1])
		    g = sns.lineplot(y='p_df', x='Composition', hue='Phase', data=ptxm_df, palette=palette, estimator=None, sort=False)
		    g.set_title("P-x at " + title)

		# Set x axis limit here if required (uncomment line below).
		g.set_xlim(0,1)
		g.set(xlabel=x_label)
		plt.savefig( "ptx_%s_%s.png" % (self.mixture[0], self.mixture[1]))

		if show_plot == True:
			plt.show()
		else:
			pass

	def ptxplot_prop_x(self,df, show_plot=False) -> None:
		""" Plot ?,x-diagram

		Use after calc_prop_x_df to plot prop,x-diagram.

		Parameters
		----------
		df : pd.DataFrame
			Dataframe obtained from calc_prop_x_df

		Returns
		-------
		None
			Save plot as png file to directory
		"""

		# User provide df obtained from calc_prop_x_df
		self.df = df

		if self.comp_to_plot == 1:
			x_label = r"$x_{%s}$" % self.mixture[0]
		elif self.comp_to_plot == 2:
			x_label = r"$x_{%s}$" % self.mixture[1]

		sns.set_style("whitegrid")
		sns.set_palette("bright")

		# Set color for saturated lines (based on sns palette).
		palette = {"saturated vapor": "C3", "saturated liquid": "C0"}

		title = str(self.set_const) + " MPa | " + str(self.mixture[0]) + "/" + str(self.mixture[1])
		g = sns.lineplot(y='prop_df', x='Composition', hue='Phase', data=self.df, palette=palette)
		g.set_title("T-x at " + title)


		# Set x axis limit here if required (uncomment line below).
		g.set_xlim(0, 1)
		g.set(xlabel=x_label)
		plt.savefig("prop-x %s_%s.png" % (self.mixture[0], self.mixture[1]))

		if show_plot == True:
			plt.show()
		else:
			pass

class PT:
	def __init__(self, env_pv: int, input_type: str, mixture: list, molanteile: list, eos_ind: object,
	             mixrule: int, path: str, unit: str, dll_path: str, p_spec: object = 0.0, t_spec: object = 0.0, fileout: str = "pt_plot.csv") -> object:
		""" Class for PT_DIAG
		Calculates points on phase envelope at constant composition for a given mixture.
		If pure fluid is set, returns the vapor pressure curve.

		Args:
			env_pv: 1 for pressure-based, 2 for volume-based during phase envelope calculation.
			input_type: set to "tliq"/"tvap"/"pliq"/"pvap". PTDIAG does not differentiate between these inputs.
			mixture: [component1, component2]
			molanteile: [molfrac1, molfrac2] in the same order as mixture.
			eos_ind: [eos_ind1, eos_ind2] in the same order as mixture, integer values for EOS type.
			mixrule: integer value for mixture rule.
			path: path to TREND's DLL folder
			unit: "molar" or "specific, but PT_DIAG only returns mol/dm3 for density.
			dll_path: path to TREND's DLL file.
			p_spec: specified pressure [MPa] for which a point on the phase envelope is calculated. No pressure is specified by default, which is 0.
			t_spec: specified temperature [K] for which a point on the phase envelope is calculated. No temperature is specified by default, which is 0.
			fileout: file name for output csv file. Default is "pt_plot.csv" in script's directory.
		"""

		self.env_pv = env_pv
		self.input_type = input_type
		self.mixture = mixture
		self.molanteile = molanteile
		self.eos_ind = eos_ind
		self.mixrule = mixrule
		self.path = path
		self.unit = unit
		self.dll_path = dll_path
		self.p_spec = p_spec
		self.t_spec = t_spec
		self.fileout = fileout

	def gen_df(self) -> pd.DataFrame:
		""" Clean up data returned from FORTRAN and return a dataframe.

		Returns:
			pd.DataFrame: dataframe with PT_DIAG results.
		"""
		
		fluid_pt = Fluid(self.input_type, self.input_type, self.mixture, self.molanteile, self.eos_ind, self.mixrule, self.path, self.unit, self.dll_path)

		ptdiag_res, point_id, ptdiag_error = fluid_pt.PTDIAG(self.env_pv, self.p_spec, self.t_spec, self.fileout)

		# Get rid of all zero values from PT_DIAG returns.
		idx_of_zero = np.where(ptdiag_res == 0)
		ptdiag_res = np.delete(ptdiag_res, idx_of_zero, axis=1)
		point_id = np.delete(point_id, idx_of_zero, axis=0)

		# Create dataframe for PT diagram
		pt_df = pd.DataFrame({
			"t_df": ptdiag_res[1],
			"p_df": ptdiag_res[0],
			"rhovap_df": ptdiag_res[2],
			"rholiq_df": ptdiag_res[3],
			"pt_id_df": point_id
		})

		#ptm_df = pt_df.melt(['t_df','p_df','pt_id_df'],value_vars=['rhovap_df','rholiq_df'],var_name='Phase',value_name='rho')

		return pt_df

	def get_df(self) -> pd.DataFrame:
		""" Return dataframe from PT_DIAG."""
		return self.gen_df()
		# TODO: this is quite redundant, consider removing.

	def ptplot(self,show_plot=True) -> None:
		""" Plot p,T-diagram

		Args:
			show_plot: if True, show plot in window.

		Returns:
			None. Save plot as png file to directory.
		"""

		pt_df = self.gen_df()
		# Initialize crit_index
		crit_index = 0

		# Set axis limits for plot. Improving readability.
		xmin = pt_df['t_df'].min()
		xmax = pt_df['t_df'].max()
		ymin = pt_df['p_df'].min()
		ymax = pt_df['p_df'].max()
		xmin_adj = xmin - 0.1 * (xmax - xmin)
		xmax_adj = xmax + 0.1 * (xmax - xmin)
		ymin_adj = ymin - 0.1 * (ymax - ymin)
		ymax_adj = ymax + 0.1 * (ymax - ymin)

		# fig = go.Figure(
		# 	data=[go.Scatter(x=pt_df['t_df'],
		# 	                 y=pt_df['p_df'],
		# 	                 mode='lines')],


		# Look for index of crit point in dataframe. Also acts as a bug-check and fail-check for point_id. Make sure plot is produced regardless if crit point is found.
		if 1 in pt_df['pt_id_df'].values:
			crit_index = pt_df.loc[pt_df['pt_id_df'] == 1].index[0]
			fig = go.Figure(
				layout={'margin':go.layout.Margin(pad=10),
			    'xaxis':{'range':[xmin_adj,xmax_adj]},
			    'yaxis':{'range':[ymin_adj,ymax_adj]}
			    }
			)
			# SV line (dashed)
			fig.add_trace(go.Scatter
				(
				x=pt_df['t_df'][:crit_index],
				y=pt_df['p_df'][:crit_index],
				mode='lines',
				name= 'Saturated Vapor',
				line=dict(color='black', dash='dash',width=3)
				)
			)
			# SL line
			fig.add_trace(go.Scatter
				(
				x=pt_df['t_df'][crit_index:],
				y=pt_df['p_df'][crit_index:],
				mode='lines',
				name= 'Saturated Liquid',
				line=dict(color='black', width=3)
			)
			)

		# Styling
		fig.update_layout(
			plot_bgcolor='white',
		)

		fig.update_xaxes(
			mirror=True,
			ticks='outside',
			showline=True,
			linecolor='black',
			gridcolor='lightgrey'
		)
		fig.update_yaxes(
			mirror=True,
			ticks='outside',
			showline=True,
			linecolor='black',
			gridcolor='lightgrey'
		)

		# Script behavior.
		if show_plot == True:
			fig.show(renderer="browser")
		else:
			pass

		fig.write_image("pt_plot.png")
	
	def calc_prop_t_df(self, prop: int) -> pd.DataFrame:
		""" Calculate prop-t on saturation lines and return dataframe.

		The structure for prop-t calculation is different from prop-x calculation. pt_df is not melted like ptx_df -> ptxm_df. propvap_df corresponds to t,rhovap and propliq_df corresponds to t,rholiq.

		Args:
			prop (int): property code for TREND_CALC. Refer to "Prop list" in TREND manual for property codes. (e.g., 32 is cvr)

		Returns:
			object: a dataframe with propvap_df and propliq_df columns added
		"""

		pt_df = self.gen_df()
		# Initialize empty lists for prop values.
		propvap = []
		propliq = []

		for t, d in zip(pt_df["t_df"], pt_df["rhovap_df"]):
			# Calctype does not matter for TREND_CALC. "cvr" is placeholder.
			fluid = Fluid("td", "cvr", self.mixture, self.molanteile, self.eos_ind, self.mixrule, self.path, self.unit, self.dll_path)
			res_tmp = fluid.TREND_CALC(t, d, 0, [prop])
			propvap.append(res_tmp[0][0])

		for t, d in zip(pt_df["t_df"], pt_df["rholiq_df"]):
			# Calctype does not matter for TREND_CALC. "cvr" is placeholder.
			fluid = Fluid("td", "cvr", self.mixture, self.molanteile, self.eos_ind, self.mixrule, self.path, self.unit, self.dll_path)
			res_tmp = fluid.TREND_CALC(t, d, 0, [prop])
			propliq.append(res_tmp[0][0])

		pt_df["propvap"] = propvap
		pt_df["propliq"] = propliq

		return pt_df

	def ptplot_prop_t(self,df,show_plot=False) -> None:
		""" Plot prop,T-diagram

		Args:
			show_plot: if True, show plot in window.

		Returns:
			None. Save plot as png file to directory.
		"""


	def multi_ptplot(self, comp_list, show_plot=False) -> None:
		""" Plot multiple p,T-diagrams on the same graph.

		Args:
			comp_list: list of compositions to plot.

		Returns:
			None. Save plot as png file to directory.
		"""
		# TODO: Does a separate plotter class make sense?

		sns.set_style("whitegrid")
		sns.set_palette("bright")

		for comp in comp_list:
			self.molanteile = [comp, 1-comp]
			pt_df = self.gen_df()
			g = sns.lineplot(x='t_df', y='p_df', data=pt_df, estimator=None, sort=False)

		plt.title("P-T Diagram")
		plt.savefig("multi_pt_plot.png")

		if show_plot == True:
			plt.show()
		else:
			pass
