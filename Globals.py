
class Globals():
    projectPath="E:/LeafDetection/mungbean"
    projectOutputPath=""
    # trayRefPoint=990 #993
    groundWeight=320

    def getPlyFileGlobals(self,file_path):
        fileInfo={}
        f = open(file_path, "rb")

        lines=f.readlines()
        file_end=0
        for line in lines:
            if file_end==1:
                break
            words = line.decode().split(" ")
            for j,word in enumerate(words):
                if word.find("end_header")>-1:
                    file_end=1
                    break
                elif word.find("marker_x_start")>-1:
                    fileInfo["marker_x_start"]=float(words[j+1][:-1])
                elif word.find("marker_y_start")>-1:
                    fileInfo["marker_y_start"]=float(words[j+1][:-1])
                elif word.find("marker_z_start")>-1:
                    fileInfo["marker_z_start"]=float(words[j+1][:-1])

                elif word.find("marker_x_stop")>-1:
                    fileInfo["marker_x_stop"]=float(words[j+1][:-1])
                elif word.find("marker_y_stop")>-1:
                    fileInfo["marker_y_stop"]=float(words[j+1][:-1])
                elif word.find("marker_z_stop")>-1:
                    fileInfo["marker_z_stop"]=float(words[j+1][:-1])

                elif word.find("field_y_origin") > -1:
                    fileInfo["field_y_origin"] = float(words[j + 1][:-1])
                elif word.find("field_z_origin") > -1:
                    fileInfo["field_z_origin"] = float(words[j + 1][:-1])
                elif word.find("field_x_origin") > -1:
                    fileInfo["field_x_origin"] = float(words[j + 1][:-1])

                elif word.find("field_x_period") > -1:
                    fileInfo["field_x_period"] = float(words[j + 1][:-1])
                elif word.find("field_y_period") > -1:
                    fileInfo["field_y_period"] = float(words[j + 1][:-1])
                elif word.find("field_z_period") > -1:
                    fileInfo["field_z_period"] = float(words[j + 1][:-1])

                elif word.find("ref_x_start") > -1:
                    fileInfo["ref_x_start"] = float(words[j + 1][:-1])
                elif word.find("ref_y_start") > -1:
                    fileInfo["ref_y_start"] = float(words[j + 1][:-1])
                elif word.find("ref_z_start") > -1:
                    fileInfo["ref_z_start"] = float(words[j + 1][:-1])

                elif word.find("ref_x_stop") > -1:
                    fileInfo["ref_x_stop"] = float(words[j + 1][:-1])
                elif word.find("ref_y_stop") > -1:
                    fileInfo["ref_y_stop"] = float(words[j + 1][:-1])
                elif word.find("ref_z_stop") > -1:
                    if words[j + 1].find("}") > -1:
                        fileInfo["ref_z_stop"] = float(words[j + 1][:-2])
                    else:
                        fileInfo["ref_z_stop"] = float(words[j + 1][:-1])
                elif word.find("y_sectors") > -1:
                    if words[j + 1].find("}") > -1:
                        fileInfo["y_sectors"] = float(words[j + 1][:-2])
                    else:
                        fileInfo["y_sectors"] = float(words[j + 1][:-1])

                elif word.find("pitch_pe") > -1:
                    fileInfo["pitch_pe"] = float(words[j + 1][:-1])

        f.close()



        return fileInfo

