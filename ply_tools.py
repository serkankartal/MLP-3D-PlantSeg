import struct
import numpy as np

class PlyFile:
    HEADER_END = b"end_header"
    
    PLY_TYPES = {
        "ushort":
            {
                "size":2,
                "pytype":'H',
                "nptype":'<u2'
            },
        "float":
            {
                "size":4,
                "pytype":'f', 
                "nptype":'f4'
            },
        "uchar":
            {
                "size":1,
                "pytype":'B', 
                "nptype":'u1'
            },
        "int":
            {
                "size":4,
                "pytype":'i',
                "nptype":'i4'
            }
    }
    
    def __init__(self):
        self.header = None
        self.file = None   
        self.header_info = {}
        self.elements = {}
        self.vertices = {}
        self.faces = None
        self.header_end_idx = 0
        self.vertex_size = 0
    
    def from_object(self, obj):
        self.elements = obj.elements
        self.vertex_size = obj.vertex_size
        for e in self.elements.keys():
            self.elements[e]["count"] = 0
    
    def get_dtypes(self):
        vertex_properties = self.elements["vertex"]["properties"]
        dtypes = []
        for vp in vertex_properties.keys():
            dtypes.append((vp, self.PLY_TYPES[vertex_properties[vp]["type"]]["nptype"]))
        return dtypes
            
    def parse_header(self):
        with open(self.file, "rb") as f:
            data = f.read()
        self.header_end_idx = data.find(self.HEADER_END)
        self.header = data[:self.header_end_idx + len(self.HEADER_END)+1].decode("utf-8")
    
    def parse_properties(self):
        current_element = None
        for line in self.header.split("\n"):
            if "obj_info" in line:
                pass
            if "element" in line:
                current_element = line.split(" ")[1]
                self.elements[current_element] = {
                    "count" : int(line.split(" ")[2]),
                    "properties" : {}
                }
            if "property" in line and current_element is not None:
                prop = line.split(" ")
                if "list" in line:
                    self.elements[current_element]["properties"][prop[4]] = {
                        "type": prop[3],
                        "size": self.PLY_TYPES[prop[3]]['size']
                    }
                else:
                    self.elements[current_element]["properties"][prop[2]] = {
                        "type": prop[1],
                        "size": self.PLY_TYPES[prop[1]]['size']
                    }
                    self.vertex_size+=self.PLY_TYPES[prop[1]]['size']
                
        #print(self.elements) 
    
    def parse_vertices(self):
        vertex_properties = self.elements["vertex"]["properties"]
        dtypes = []
        for vp in vertex_properties.keys():
            dtypes.append((vp, self.PLY_TYPES[vertex_properties[vp]["type"]]["nptype"]))
        
        self.vertices = np.zeros((self.elements["vertex"]["count"]), dtype=dtypes)
        data_idx = self.header_end_idx + len(self.HEADER_END)+1
        with open(self.file, "rb") as f:
            data = f.read()
        bin_data = data[data_idx:]
        
        cursor = 0
        
        for vi in range(0, self.elements["vertex"]["count"]):
            for vpi, vprop in enumerate(vertex_properties.keys()):
                self.vertices[vi][vpi] = struct.unpack(self.PLY_TYPES[vertex_properties[vprop]['type']]['pytype'], bin_data[cursor:cursor+vertex_properties[vprop]["size"]])[0]
                cursor += vertex_properties[vprop]["size"]
    
    def parse_faces(self):
        if 'face' not in self.elements.keys() or self.elements["face"]["count"] == 0:
            return
        self.faces = np.zeros((self.elements["face"]["count"], 3), dtype='i4')
        data_idx = self.header_end_idx + len(self.HEADER_END)+1 + self.vertex_size * self.elements["vertex"]["count"]
        with open(self.file, "rb") as f:
            data = f.read()
        bin_data = data[data_idx:]
        
        for fidx in range(0, self.elements["face"]["count"]):
            for i in range(0, 3):
                didx = fidx * 13 + i * 4
                self.faces[fidx][i] = struct.unpack('i', bin_data[didx+1:didx+5])[0]
    
    def load(self, filename):
        self.file = filename
        self.parse_header()
        self.parse_properties()
        self.parse_vertices()
        self.parse_faces()
    
    def write(self, filename):
        # self.build_header()
        with open(filename, 'w') as f:
            f.write(self.header)
        if self.vertices is not None:
            with open(filename, 'ab') as f:
                vertex_properties = self.elements["vertex"]["properties"]
                for vertex in self.vertices:
                    for vi, vprop in enumerate(vertex_properties.keys()):
                        pval = struct.pack(self.PLY_TYPES[vertex_properties[vprop]["type"]]["pytype"], vertex[vi])
                        f.write(pval)
        if self.faces is not None:
            with open(filename, 'ab') as f:
                for face in self.faces:
                    flen = 3
                    f.write(struct.pack('B', flen))
                    for vf in face:
                        f.write(struct.pack('i', vf))
    
    def get_vertex_pos_matrix(self):    
        m = np.zeros((self.vertices.shape[0], 3), dtype='f4')
        m[:, 0] = self.vertices['x']
        m[:, 1] = self.vertices['y']
        m[:, 2] = self.vertices['z']
        return m
    
    def get_vertex_color_matrix(self, norm=False):
        
        if not norm:
            m = np.zeros((self.vertices.shape[0], 3), dtype='u1')
            m[:, 0] = self.vertices['red']
            m[:, 1] = self.vertices['green']
            m[:, 2] = self.vertices['blue']
        else:
            m = np.zeros((self.vertices.shape[0], 3), dtype='f4')
            m[:, 0] = self.vertices['red']
            m[:, 1] = self.vertices['green']
            m[:, 2] = self.vertices['blue']
        return m / 255.0

    def build_header(self):
        h = "ply\n"
        h += "format binary_little_endian 1.0\n"
        h += "comment PLY file generated by the Andras\n"
        
        h += "element vertex {}\n".format(self.elements["vertex"]["count"])
        vertex_properties = self.elements["vertex"]["properties"]
        for vprop in vertex_properties.keys():
            h += "property {} {}\n".format(vertex_properties[vprop]["type"], vprop)
    
        h += "element face {}\n".format(self.elements["face"]["count"])
        h += "property list uchar int vertex_index\n\n"
        h += "end_header\n"
        self.header = h
    
    def add_faces(self, faces):
        self.faces = faces
        if "face" not in self.elements.keys():
            self.elements["face"] = {}
        self.elements["face"]["count"] = faces.shape[0]