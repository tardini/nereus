import logging
from xml.etree import cElementTree as ET
from xml.dom import minidom

fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s: %(message)s', '%H:%M:%S')
hnd = logging.StreamHandler()
hnd.setFormatter(fmt)
logger = logging.getLogger('XML')
logger.addHandler(hnd)
logger.setLevel(logging.INFO)

from collections import defaultdict
try:
    basestring
except NameError:  # python3
    basestring = str


def dict_to_etree(d):

    def _to_etree(d, root):

        if not d:
            pass
        elif isinstance(d, basestring):
            root.text = d
        elif isinstance(d, bool):
            if d:
                root.text = 'true'
            else:
                root.text = 'false'
        elif isinstance(d, dict):
            for k,v in d.items():
                assert isinstance(k, basestring)
                if k.startswith('#'):
                    assert k == '#text' and isinstance(v, basestring)
                    root.text = v
                elif k.startswith('@'):
                    assert isinstance(v, basestring)
                    root.set(k[1:], v)
                elif isinstance(v, list):
                    for e in v:
                        _to_etree(e, ET.SubElement(root, k))
                else:
                    _to_etree(v, ET.SubElement(root, k))
        else:
            raise TypeError('invalid type: ' + str(type(d)))
    assert isinstance(d, dict) and len(d) == 1
    tag, body = next(iter(d.items()))
    node = ET.Element(tag)
    _to_etree(body, node)

    return ET.tostring(node)


def etree_to_dict(t):

    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k:v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v) for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
              d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d


def xml2dict(f_xml):
    with open(f_xml, 'r') as f:
        xml_str = f.read()
    etree = ET.XML(xml_str)
    return etree_to_dict(etree)


def dict2xml(xml_d, f_xml):

    xml_str = dict_to_etree(xml_d)
    xdom = minidom.parseString(xml_str)
    xml_pretty = xdom.toprettyxml(indent='    ')

    with open(f_xml, 'w') as f:
        f.write(xml_pretty)


def xml2val_node(xml_node):

    setup_node = {}
    for key, val_d in xml_node.items():
        if val_d['@type'] == 'str':
            if '#text' in val_d.keys():
                setup_node[key] = val_d['#text']
            else:
                setup_node[key] = ''
        elif val_d['@type'] == 'bool':
            if val_d['#text'].lower() == 'true':
                setup_node[key] = True
            else:
                setup_node[key] = False
        elif val_d['@type'] == 'int':
            setup_node[key] = int(val_d['#text'])
        elif val_d['@type'] == 'flt':
            setup_node[key] = float(val_d['#text'])

    return setup_node


def xml2val_dic(xml_d):

    setup_d = {}
    for node, xml_node in xml_d.items():
        setup_d[node] = xml2val_node(xml_node)
    return setup_d


if __name__ == '__main__':

    fxml = 'deep.xml'
    fxml_out = 'deep_out.xml'

    c = xml2dict(fxml)
    for key, val in c.items():
        print(key)
        for key2, val2 in val.items():
            print(key2, val2)
    dict2xml(c, fxml_out)
