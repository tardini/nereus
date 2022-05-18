import logging
from xml.dom import minidom

fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s: %(message)s', '%H:%M:%S')
hnd = logging.StreamHandler()
hnd.setFormatter(fmt)
logger = logging.getLogger('XML')
logger.addHandler(hnd)
logger.setLevel(logging.INFO)


class DIX:


    def dict2xml(self, setdict, f_xml):

        doc = minidom.Document()
        input = doc.createElement('input')
        doc.appendChild(input)

        for key, val in setdict.items():
            var = doc.createElement(key)
            input.appendChild(var)
            if type(val) is dict:
                for val_key, val_val in val.items():
                    val_var = doc.createElement(val_key)
                    var.appendChild(val_var)
                    ptext = doc.createTextNode(val_val.strip())
                    val_var.appendChild(ptext)
            else:
                if type(val) is bool:
                    if val:
                        val2 = 'true'
                    else:
                        val2 = 'false'
                else:
                    val2 = str(val)
                ptext = doc.createTextNode(val2.strip())
                var.appendChild(ptext)

        file = open(f_xml, 'wb')
        file.write(doc.toprettyxml(indent='  ', newl='\n', encoding='UTF-8'))
        file.close()
        logger.info('Saved setup to XML file %s' %f_xml)


    def xml2dict(self, f_xml):

        logger.info('Loading setup from XML file %s' %f_xml)
        doc = minidom.parse(f_xml)
        node = doc.documentElement
        dinit = {}
        for child in node.childNodes:
            if child.attributes is not None:
                tag = child.tagName.strip()
                if child.firstChild is not None:
                    dinit[tag] = child.firstChild.data.strip()

        return dinit
