def generate_tile_crd_glb_matmul(struct, sizes_dict_level0):
    assert struct is not None
    glb_arr = {"B_seg0": [0], "B_crd0": [], "B_seg1": [0],
               "B_crd1": [], "C_seg0": [0], "C_crd0": [],
               "C_seg1": [0], "C_crd1": []}
    ref_glb_convertor = {"B": [], "C": []}
    for i in range(struct["i00"]):
        for j in range(struct["k00"]):
            if (i, j) in sizes_dict_level0["B"]:
                glb_arr["B_crd1"].append(j)
                ref_glb_convertor["B"].append(str(i) + "_" + str(j))
        glb_arr["B_seg1"].append(len(glb_arr["B_crd1"]))
        glb_arr["B_crd0"].append(i)
    glb_arr["B_seg0"].append(len(glb_arr["B_crd0"]))

    cnt_i = 0
    for i in range(struct["k00"]):
        for j in range(struct["j00"]):
            if (i, j) in sizes_dict_level0["C"]:
                glb_arr["C_crd1"].append(j)
                ref_glb_convertor["C"].append(str(i) + "_" + str(j))
        glb_arr["C_seg1"].append(len(glb_arr["C_crd1"]))
        glb_arr["C_crd0"].append(i)
    glb_arr["C_seg0"].append(len(glb_arr["C_crd0"]))
    return ref_glb_convertor, glb_arr


def generate_tile_crd_mem_matmul(struct, sizes_dict_level1, key_tokens,
                                 ref_glb_convertor, ref_to_crd_convertor):
    assert struct is not None
    if isinstance(key_tokens[0], int) and isinstance(key_tokens[1], int):
        B_token = ref_glb_convertor["B"][key_tokens[0]].split("_")
        C_token = ref_glb_convertor["C"][key_tokens[1]].split("_")
        B_k00_ = int(B_token[1])
        B_i00_ = int(B_token[0])
        C_j00_ = int(C_token[1])
        C_k00_ = int(C_token[0])
    mem_arr = {"B_seg0": [0], "B_crd0": [], "B_seg1": [0],
               "B_crd1": [], "C_seg0": [0], "C_crd0": [], "C_seg1": [0], "C_crd1": []}
    key1 = "B_" + str(key_tokens[0])
    key2 = "C_" + str(key_tokens[1])
    k_ = []
    for i in range(struct["i0"]):
        fl = False
        for j in range(struct["k0"]):
            if (B_i00_, B_k00_, i, j) in sizes_dict_level1["B"]:
                mem_arr["B_crd1"].append(j)
                fl = True
                k_.append(str(i) + "_" + str(j))
        if fl:
            mem_arr["B_seg1"].append(len(mem_arr["B_crd1"]))
            mem_arr["B_crd0"].append(i)
    mem_arr["B_seg0"].append(len(mem_arr["B_crd0"]))
    ref_to_crd_convertor[key1] = k_
    k_ = []
    for i in range(struct["k0"]):
        fl = False
        for j in range(struct["j0"]):
            if (C_k00_, C_j00_, i, j) in sizes_dict_level1["C"]:
                mem_arr["C_crd1"].append(j)
                k_.append(str(i) + "_" + str(j))
                fl = True
        if fl:
            mem_arr["C_seg1"].append(len(mem_arr["C_crd1"]))
            mem_arr["C_crd0"].append(i)
    mem_arr["C_seg0"].append(len(mem_arr["C_crd0"]))
    ref_to_crd_convertor[key2] = k_
    print_mem_arr = []
    for i_ in sizes_dict_level1["C"].keys():
        if i_[0] == C_k00_ and i_[1] == C_j00_:
            print_mem_arr.append(str(i_[2]) + "_" + str(i_[3]))
    assert len(print_mem_arr) == len(mem_arr["C_crd1"])
    return ref_to_crd_convertor, mem_arr
