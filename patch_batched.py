from pathlib import Path
r=Path('/home/saeed/glm-tuning-20261011/llama/ggml/src/ggml-cpu/amx')
p=r/'amx.cpp';s=p.read_text();s=s.replace('''        if (qtype_has_amx_kernels(src0->type) && ((!moe && src0->ne[2] != 1) || src0->ne[3] != 1)) {
            return false;
        }
''','');s=s.replace('''            default:
                return false;
        }
        if (src0->ne[0] % alignment)''','''            case GGML_TYPE_F16:
                alignment = 16;
                break;
            default:
                return false;
        }
        if (src0->ne[0] % alignment)''');p.write_text(s)
p=r/'mmq.cpp';s=p.read_text();s=s.replace('(const float *)src1->data + src1_offset + mb_start * K','(const float *)((const char *)src1->data + src1_offset) + mb_start * K').replace('(const type *)src0->data + src0_offset + nb_start * K','(const type *)((const char *)src0->data + src0_offset) + nb_start * K').replace('(float *)dst->data + dst_offset + mb_start * ldc + nb_start','(float *)((char *)dst->data + dst_offset) + mb_start * ldc + nb_start').replace('(float *) dst->data + dst_offset + nb_start','(float *) ((char *) dst->data + dst_offset) + nb_start').replace('(float *) dst->data + dst_offset + mb_start * N + nb_start','(float *) ((char *) dst->data + dst_offset) + mb_start * N + nb_start')
old='int64_t src0_offset = ggml_batch_offset(src0, batch_idx, ne2);'
first=s.index(old);# floating path uses ordinary tensor strides and broadcasting
s=s[:first]+s[first:].replace(old,'''int64_t src0_batch = (batch_idx % ne2) / (ne2 / src0->ne[2]) +
                        (batch_idx / ne2) / (dst->ne[3] / src0->ne[3]) * src0->ne[2];
                    int64_t src0_offset = ggml_batch_offset(src0, src0_batch, src0->ne[2]);''',1)
s=s.replace(old,'''int64_t src0_batch = (batch_idx % ne2) / (ne2 / src0->ne[2]) +
                        (batch_idx / ne2) / (dst->ne[3] / src0->ne[3]) * src0->ne[2];
                    int64_t src0_offset = src0_batch * N * get_row_size<type, blck_size>(K);''')
p.write_text(s)
