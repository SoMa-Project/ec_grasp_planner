from xml.dom import minidom
import numpy as np

import components
import utils

def print(xml_string):
    # pretty print
    print minidom.parseString(xml_string).toprettyxml()

def xml2ha(xml_string):
    xmldoc = minidom.parseString(xml_string)
    
    ha_list = xmldoc.getElementsByTagName('HybridAutomaton')
    assert(len(ha_list) == 1)  # ensure that there is exactly ONE hybrid automaton defined
    ha = ha_list[0]
    
    hybrid_automaton = components.HybridAutomaton(**{i.name: i.value for i in ha.attributes.values()})
    
    # iterate through control modes
    for cm in ha.getElementsByTagName('ControlMode'):
        control_mode = components.ControlMode(**{i.name: i.value for i in cm.attributes.values()})
        
        # get control set
        cs_list = cm.getElementsByTagName('ControlSet')
        assert(len(cs_list) == 1)
        cs = cs_list[0]
        
        props = {i.name: i.value for i in cs.attributes.values()}
        try:
            control_set = components.ControlSet(**props)
        except TypeError as e:
            additional_props = {'js_kd': props['js_kd'], 'js_kp': props['js_kp']}
            del props['js_kd']
            del props['js_kp']
            control_set = components.ControlSet(**props)
            control_set.properties.update(additional_props)
        
        # iterate through controllers
        for cntrl in cs.getElementsByTagName('Controller'):
            cntrl_type = cntrl.attributes['type'].value
            
            props = utils.numpify_dict({i.name: i.value for i in cntrl.attributes.values()})
            #del props['type']
            #print cntrl_type
            #del props['priority']
            #control_set.add(components.__dict__[cntrl_type](**props))
            
            additional_props = {}
            
            while True:
                try:
                    cntrl = components.Controller(**props)
                    cntrl.properties.update(additional_props)
                    break
                except TypeError as e:
                    unexpected_keyword_argument = e.message[e.message.find('\'')+1:-1]
                    additional_props[unexpected_keyword_argument] = props[unexpected_keyword_argument]
                    del props[unexpected_keyword_argument]
            
            control_set.add(cntrl)
        
        control_mode.set(control_set)
        hybrid_automaton.add(control_mode)
    
    # iterate through control switches
    for cs in ha.getElementsByTagName('ControlSwitch'):
        control_switch = components.ControlSwitch(**{i.name: i.value for i in cs.attributes.values()})
        
        # get jump conditions
        for jc in cs.getElementsByTagName('JumpCondition'):
            props = {i.name: i.value for i in jc.attributes.values()}
            
            # get sensor -- assume that there's just one
            sensor = jc.getElementsByTagName('Sensor')[0]
            sensor_props = {i.name: i.value for i in sensor.attributes.values()}
            sensor_props['sensor_type'] = sensor_props['type']
            del sensor_props['type']
            props.update(sensor_props)
            
            props = utils.numpify_dict(props)
            
            jump = components.JumpCondition(**props)
            control_switch.add(jump)
        
        hybrid_automaton.add(control_switch)
    
    return hybrid_automaton
    
