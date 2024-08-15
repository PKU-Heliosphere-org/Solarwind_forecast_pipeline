# GONG, PFSS and torch dependencies: modules, constants and functions
import sunpy.map
import pfsspy
from sunpy.sun import constants
import astropy.units as u
import numpy as np

nrho = 30
rss = 2.5

def get_gong_map_obj(path_to_file):
    # create gong map object
    gmap_temp = sunpy.map.Map(path_to_file)
    gmap = sunpy.map.Map(gmap_temp.data - np.mean(gmap_temp.data), gmap_temp.meta)

    # added by RONG 2024/8/15, to prevent some 
    gmap.meta['rsun_ref'] = constants.radius / u.m
    gmap.meta['bunit'] = 'G'
    return gmap

def get_ss_graphs(df, i_df):
    col = df.iloc[i_df]
    
    gong_map_3 = get_gong_map_obj(col['path_3_days_ago'])
    input_3 = pfsspy.Input(gong_map_3, nrho, rss)
    output_3 = pfsspy.pfss(input_3)
    graph_3 = output_3.source_surface_br.data
    lat_3 = int(gong_map_3.carrington_latitude.to_value().round())
    sub_graph_3 = graph_3[lat_3+90-45:lat_3+90+45,90:-90]
    
    gong_map_4 = get_gong_map_obj(col['path_4_days_ago'])
    input_4 = pfsspy.Input(gong_map_4, nrho, rss)
    output_4 = pfsspy.pfss(input_4)
    graph_4 = output_4.source_surface_br.data
    lat_4 = int(gong_map_4.carrington_latitude.to_value().round())
    sub_graph_4 = graph_4[lat_4+90-45:lat_4+90+45,90:-90]
    
    gong_map_5 = get_gong_map_obj(col['path_5_days_ago'])
    input_5 = pfsspy.Input(gong_map_5, nrho, rss)
    output_5 = pfsspy.pfss(input_5)
    graph_5 = output_5.source_surface_br.data
    lat_5 = int(gong_map_5.carrington_latitude.to_value().round())
    sub_graph_5 = graph_5[lat_5+90-45:lat_5+90+45,90:-90]
    
    gong_map_6 = get_gong_map_obj(col['path_6_days_ago'])
    input_6 = pfsspy.Input(gong_map_6, nrho, rss)
    output_6 = pfsspy.pfss(input_6)
    graph_6 = output_6.source_surface_br.data
    lat_6 = int(gong_map_6.carrington_latitude.to_value().round())
    sub_graph_6 = graph_6[lat_6+90-45:lat_6+90+45,90:-90]

    ss_graphs = np.array([sub_graph_6,sub_graph_5,sub_graph_4,sub_graph_3])
    
    return ss_graphs