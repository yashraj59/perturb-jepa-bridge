# F103 PerturbMulti RNA Obs and Pairing Preflight

## Decision
`F103_PERTURBMULTI_RNA_OBS_AND_PAIRING_PREFLIGHT_PASS_READY_FOR_SEALED_F082_VALIDATION_DESIGN`

No model is promoted. No new architecture is trained. The protected rank-3 train-split-only PLS raw-linear readout remains the model of record.

## Scope
- candidate: PerturbMulti CRISPR screen
- model path under protection: frozen F082/F096 ProgramBootstrapJEPA path
- checks run: Hugging Face manifest, RNA/protein H5AD obs-only HDF5 reads, image tar header range samples
- checks not run: `.X` matrix loading, image payload extraction, model fitting, calibration, or promotion

## H5AD Obs Probe
label	path	index_key	n_obs	n_vars	obs_columns	required_obs_columns_present	missing_required_obs_columns	split_usable_columns_present	id_alias_columns_present	perturbation_alias_columns_present	coordinate_alias_columns_present	contract_usable	cell_id_unique	cell_id_head	cell_name_unique	cell_name_head	condition_unique	condition_head	perturbation_unique	perturbation_head	fov_unique	fov_head	x_unique	x_head	y_unique	y_head	z_unique	z_head	dataset_unique	dataset_head
rna	/content/hf_cache/datasets--xingjiepan--PerturbMulti/snapshots/8aac954eb631b68f6e11171a8313db61cc16c38c/RNA_scaled_crispr_screen_20240615.h5ad	id	2206191	209	fov,x,y,global_x,global_y,area,batch,cell_type,cluster_type,singlet_name,singlet_gene,bc3,bc1,n_thresh3,n_thresh1	False	cell_id,cell_name,condition,dataset,perturbation,z	fov	id,_index	singlet_gene,singlet_name,bc1,bc3	fov,x,y,global_x,global_y	True	0		0		0		0		1147	138,604,61,221,512	2206041	-2720.3153415370466,5843.99005344489,-4031.937240120458,-1570.4826412845543,1765.691048670628	2206077	5288.399633599112,5577.074527725294,7263.2725798472975,2084.2767868079004,7085.6163107675375	0		0	
protein	/content/hf_cache/hub/datasets--xingjiepan--PerturbMulti/snapshots/8aac954eb631b68f6e11171a8313db61cc16c38c/protein_intensities_crispr_screen_20240615.h5ad	_index	99294	18	cell_id,condition,cell_name,fov,x,y,z,area,perturbation,top_bc_counts,dataset	True		dataset,fov,cell_name,perturbation	cell_id,_index,cell_name	perturbation	fov,x,y,z	True	99294	0,1,2,3,4	99294	155844328857198891569767177470408245589,87037960057500566910344714542817530155,275154305730943958138352683642239782165,327046536077761088987800153824495004070,85862599036424094053762788041439828612	1	adlib,adlib,adlib,adlib,adlib	508	Pten_gBC_00236,Snip1_gBC_00359,Mtor_gBC_00124,Sf3a2_gBC_00122,Sf3b6_gBC_00088	1129	939,335,518,649,899	99294	3922.2921612573637,-3701.9815633577364,206.25628671180087,1083.347798683804,4371.968540395763	99293	-5031.289005293578,9487.276679620352,1174.6480643016437,3334.6291749905245,11928.290613651336	10	7,0,0,0,0	11	IE48_240531_L7_cas9_proteins_PL109_PL112_M6.zarr,IE46_240529_cas9_L7_40X_protein_PL109_PL112_batch2.zarr,IE43_240501_L7_cas9_proteins_PL109_PL112_M6.zarr,IE42_240429_L7_cas9_proteins_PL109_PL112_M6.zarr,IE47_240531_cas9_L7_40X_protein_PL109_PL112_batch2.zarr

## Pairing Overlap
rna_cell_id_count	protein_cell_id_count	rna_alias_id_count	protein_alias_id_count	image_sample_id_count	rna_protein_cell_id_overlap	rna_protein_cell_id_overlap_fraction_of_rna	rna_protein_cell_id_overlap_fraction_of_protein	image_rna_alias_overlap	image_protein_alias_overlap	image_rna_protein_alias_overlap	rna_protein_examples	image_rna_examples	image_protein_examples
2206191	99294	2206191	198588	500	93848	0.04253847468328898	0.9451527786170363	281	17	17	100001212342975985766672256078858246231,100001546057789672196375941637678957774,10000372391912158512952532434829519247,100004686964922353514544703478806832257,10001061918926768954970977187147232	100000235360096190881752765667944812208,100000251864987355764392889221243415948,100000364886166220073588621151611719403,100000489918494630920482672582203866452,100000830185540017450374120249317817068	100001212342975985766672256078858246231,100001546057789672196375941637678957774,10000372391912158512952532434829519247,100004686964922353514544703478806832257,10001061918926768954970977187147232

## Image Tar Header Sample
tar_file	member_name	normalized_image_id	size_bytes	typeflag	offset
crispr_screen_20240615_chunk_aa.tar	100000024761494857298569926110060627021.npz	100000024761494857298569926110060627021	78611	0	0
crispr_screen_20240615_chunk_aa.tar	100000235360096190881752765667944812208.npz	100000235360096190881752765667944812208	94424	0	79360
crispr_screen_20240615_chunk_aa.tar	100000238608830404563000561053135801800.npz	100000238608830404563000561053135801800	39305	0	174592
crispr_screen_20240615_chunk_aa.tar	100000251864987355764392889221243415948.npz	100000251864987355764392889221243415948	131842	0	214528
crispr_screen_20240615_chunk_aa.tar	100000364886166220073588621151611719403.npz	100000364886166220073588621151611719403	146901	0	347136
crispr_screen_20240615_chunk_aa.tar	100000489918494630920482672582203866452.npz	100000489918494630920482672582203866452	87602	0	494592
crispr_screen_20240615_chunk_aa.tar	100000830185540017450374120249317817068.npz	100000830185540017450374120249317817068	67630	0	583168
crispr_screen_20240615_chunk_aa.tar	100000943623197467400129996971617948926.npz	100000943623197467400129996971617948926	146320	0	651776
crispr_screen_20240615_chunk_aa.tar	100001002088337103064787413928026615920.npz	100001002088337103064787413928026615920	52242	0	798720
crispr_screen_20240615_chunk_aa.tar	100001212342975985766672256078858246231.npz	100001212342975985766672256078858246231	132698	0	851968
crispr_screen_20240615_chunk_aa.tar	100001414326317306961335389448996832749.npz	100001414326317306961335389448996832749	36421	0	985600
crispr_screen_20240615_chunk_aa.tar	100001546057789672196375941637678957774.npz	100001546057789672196375941637678957774	265143	0	1022976
crispr_screen_20240615_chunk_aa.tar	100001686499937634276896147819564186281.npz	100001686499937634276896147819564186281	32364	0	1288704
crispr_screen_20240615_chunk_aa.tar	10000193181701322695309738112087928187.npz	10000193181701322695309738112087928187	65831	0	1321984
crispr_screen_20240615_chunk_aa.tar	100001942800326932781211362742756254304.npz	100001942800326932781211362742756254304	42235	0	1388544
crispr_screen_20240615_chunk_aa.tar	100002086081481261348637268776090156999.npz	100002086081481261348637268776090156999	49244	0	1431552
crispr_screen_20240615_chunk_aa.tar	100002158983656557713946336365906233077.npz	100002158983656557713946336365906233077	49212	0	1481728
crispr_screen_20240615_chunk_aa.tar	100002171017808260700432409854902357432.npz	100002171017808260700432409854902357432	52503	0	1531904
crispr_screen_20240615_chunk_aa.tar	100002397297103439076661068531401716607.npz	100002397297103439076661068531401716607	137784	0	1585152
crispr_screen_20240615_chunk_aa.tar	10000240839447879029459481488857114886.npz	10000240839447879029459481488857114886	131470	0	1723904

## Relevant Manifest Rows
path	size_bytes	is_crispr_rna	is_crispr_protein	is_crispr_image_tar
RNA_scaled_crispr_screen_20240615.h5ad		True	False	False
crispr_screen_20240615_chunk_aa.tar		False	False	True
crispr_screen_20240615_chunk_ab.tar		False	False	True
crispr_screen_20240615_chunk_ac.tar		False	False	True
crispr_screen_20240615_chunk_ad.tar		False	False	True
crispr_screen_20240615_chunk_ae.tar		False	False	True
crispr_screen_20240615_chunk_af.tar		False	False	True
crispr_screen_20240615_chunk_ag.tar		False	False	True
crispr_screen_20240615_chunk_ah.tar		False	False	True
crispr_screen_20240615_chunk_ai.tar		False	False	True
crispr_screen_20240615_chunk_aj.tar		False	False	True
crispr_screen_20240615_chunk_ak.tar		False	False	True
crispr_screen_20240615_chunk_al.tar		False	False	True
crispr_screen_20240615_chunk_am.tar		False	False	True
crispr_screen_20240615_chunk_an.tar		False	False	True
crispr_screen_20240615_chunk_ao.tar		False	False	True

## Raw Data Handling
- RNA H5AD local path: `/content/hf_cache/datasets--xingjiepan--PerturbMulti/snapshots/8aac954eb631b68f6e11171a8313db61cc16c38c/RNA_scaled_crispr_screen_20240615.h5ad`
- Protein H5AD local path: `/content/hf_cache/hub/datasets--xingjiepan--PerturbMulti/snapshots/8aac954eb631b68f6e11171a8313db61cc16c38c/protein_intensities_crispr_screen_20240615.h5ad`
- raw payloads are outside git under the Hugging Face cache

## Next Step
If this preflight passes, design a small sealed PerturbMulti F082 validation run on GPU. Keep PLS/full-ridge as the protected audit floor only and do not promote without a fresh external Tier 3 pass.

## Machine-Readable Summary
```json
{
  "decision": "F103_PERTURBMULTI_RNA_OBS_AND_PAIRING_PREFLIGHT_PASS_READY_FOR_SEALED_F082_VALIDATION_DESIGN",
  "downloads": {
    "RNA_scaled_crispr_screen_20240615.h5ad": {
      "existed_before": true,
      "exists": true,
      "local_path": "/content/hf_cache/datasets--xingjiepan--PerturbMulti/snapshots/8aac954eb631b68f6e11171a8313db61cc16c38c/RNA_scaled_crispr_screen_20240615.h5ad",
      "size_bytes": 14201294683
    },
    "protein_intensities_crispr_screen_20240615.h5ad": {
      "existed_before": true,
      "exists": true,
      "local_path": "/content/hf_cache/hub/datasets--xingjiepan--PerturbMulti/snapshots/8aac954eb631b68f6e11171a8313db61cc16c38c/protein_intensities_crispr_screen_20240615.h5ad",
      "size_bytes": 31430488
    }
  },
  "environment": {
    "cuda_visible_devices": "",
    "cwd": "/content/perturb-jepa-bridge",
    "hf_home": "/content/hf_cache",
    "pid": 51439
  },
  "image_tar_errors": [],
  "model_promoted": false,
  "model_trained": false,
  "notes": "HDF5 obs-only H5AD reads and HTTP range-read tar headers only; no .X matrix reads and no model fitting.",
  "overlap": {
    "image_protein_alias_overlap": 17,
    "image_protein_examples": "100001212342975985766672256078858246231,100001546057789672196375941637678957774,10000372391912158512952532434829519247,100004686964922353514544703478806832257,10001061918926768954970977187147232",
    "image_rna_alias_overlap": 281,
    "image_rna_examples": "100000235360096190881752765667944812208,100000251864987355764392889221243415948,100000364886166220073588621151611719403,100000489918494630920482672582203866452,100000830185540017450374120249317817068",
    "image_rna_protein_alias_overlap": 17,
    "image_sample_id_count": 500,
    "protein_alias_id_count": 198588,
    "protein_cell_id_count": 99294,
    "rna_alias_id_count": 2206191,
    "rna_cell_id_count": 2206191,
    "rna_protein_cell_id_overlap": 93848,
    "rna_protein_cell_id_overlap_fraction_of_protein": 0.9451527786170363,
    "rna_protein_cell_id_overlap_fraction_of_rna": 0.04253847468328898,
    "rna_protein_examples": "100001212342975985766672256078858246231,100001546057789672196375941637678957774,10000372391912158512952532434829519247,100004686964922353514544703478806832257,10001061918926768954970977187147232"
  },
  "preflight_pass": true,
  "protein_summary": {
    "cell_id_head": "0,1,2,3,4",
    "cell_id_unique": 99294,
    "cell_name_head": "155844328857198891569767177470408245589,87037960057500566910344714542817530155,275154305730943958138352683642239782165,327046536077761088987800153824495004070,85862599036424094053762788041439828612",
    "cell_name_unique": 99294,
    "condition_head": "adlib,adlib,adlib,adlib,adlib",
    "condition_unique": 1,
    "contract_usable": true,
    "coordinate_alias_columns_present": "fov,x,y,z",
    "dataset_head": "IE48_240531_L7_cas9_proteins_PL109_PL112_M6.zarr,IE46_240529_cas9_L7_40X_protein_PL109_PL112_batch2.zarr,IE43_240501_L7_cas9_proteins_PL109_PL112_M6.zarr,IE42_240429_L7_cas9_proteins_PL109_PL112_M6.zarr,IE47_240531_cas9_L7_40X_protein_PL109_PL112_batch2.zarr",
    "dataset_unique": 11,
    "fov_head": "939,335,518,649,899",
    "fov_unique": 1129,
    "id_alias_columns_present": "cell_id,_index,cell_name",
    "index_key": "_index",
    "label": "protein",
    "missing_required_obs_columns": "",
    "n_obs": 99294,
    "n_vars": 18,
    "obs_columns": "cell_id,condition,cell_name,fov,x,y,z,area,perturbation,top_bc_counts,dataset",
    "path": "/content/hf_cache/hub/datasets--xingjiepan--PerturbMulti/snapshots/8aac954eb631b68f6e11171a8313db61cc16c38c/protein_intensities_crispr_screen_20240615.h5ad",
    "perturbation_alias_columns_present": "perturbation",
    "perturbation_head": "Pten_gBC_00236,Snip1_gBC_00359,Mtor_gBC_00124,Sf3a2_gBC_00122,Sf3b6_gBC_00088",
    "perturbation_unique": 508,
    "required_obs_columns_present": true,
    "split_usable_columns_present": "dataset,fov,cell_name,perturbation",
    "x_head": "3922.2921612573637,-3701.9815633577364,206.25628671180087,1083.347798683804,4371.968540395763",
    "x_unique": 99294,
    "y_head": "-5031.289005293578,9487.276679620352,1174.6480643016437,3334.6291749905245,11928.290613651336",
    "y_unique": 99293,
    "z_head": "7,0,0,0,0",
    "z_unique": 10
  },
  "raw_data_outside_git": true,
  "rna_summary": {
    "cell_id_head": "",
    "cell_id_unique": 0,
    "cell_name_head": "",
    "cell_name_unique": 0,
    "condition_head": "",
    "condition_unique": 0,
    "contract_usable": true,
    "coordinate_alias_columns_present": "fov,x,y,global_x,global_y",
    "dataset_head": "",
    "dataset_unique": 0,
    "fov_head": "138,604,61,221,512",
    "fov_unique": 1147,
    "id_alias_columns_present": "id,_index",
    "index_key": "id",
    "label": "rna",
    "missing_required_obs_columns": "cell_id,cell_name,condition,dataset,perturbation,z",
    "n_obs": 2206191,
    "n_vars": 209,
    "obs_columns": "fov,x,y,global_x,global_y,area,batch,cell_type,cluster_type,singlet_name,singlet_gene,bc3,bc1,n_thresh3,n_thresh1",
    "path": "/content/hf_cache/datasets--xingjiepan--PerturbMulti/snapshots/8aac954eb631b68f6e11171a8313db61cc16c38c/RNA_scaled_crispr_screen_20240615.h5ad",
    "perturbation_alias_columns_present": "singlet_gene,singlet_name,bc1,bc3",
    "perturbation_head": "",
    "perturbation_unique": 0,
    "required_obs_columns_present": false,
    "split_usable_columns_present": "fov",
    "x_head": "-2720.3153415370466,5843.99005344489,-4031.937240120458,-1570.4826412845543,1765.691048670628",
    "x_unique": 2206041,
    "y_head": "5288.399633599112,5577.074527725294,7263.2725798472975,2084.2767868079004,7085.6163107675375",
    "y_unique": 2206077,
    "z_head": "",
    "z_unique": 0
  }
}
```
