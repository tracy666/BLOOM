"""Exact metric definitions and decomposition support, independent of plotting."""
TRANS = ['Autoformer','Crossformer','DLinear','FEDformer','FiLM','FreTS','Informer','iTransformer','LightTS','Mamba','Nonstationary_Transformer','PatchTST','PAttn','Reformer','SegRNN','TiDE','TimeFilter','TimesNet','TimeXer','Transformer','TSMixer','WPMixer']
METRICS = {}
def add(names, direction, aspect, day=True, gene=False, recommended=False, eligible=True, note=''):
    for name in names.split():
        METRICS[name] = dict(metric=name, direction=direction, aspect=aspect, per_day=day, per_gene=gene, recommended=recommended, benchmark_eligible=eligible, note=note)
add('train_log_mean_day_pcc_cell mean_day_pcc_cell mean_cell_pcc median_cell_pcc', 'higher', 'cell expression profile', gene=False, recommended=True, note='Requires paired target cells. Global training means weight days equally; pooled raw mean weights cells.')
add('train_log_day_pcc_cell','higher','cell expression profile',eligible=False,note='Daily log values; no gene decomposition of a correlation across genes.')
add('train_log_mean_day_pcc_gene mean_day_pcc_gene mean_gene_pcc median_gene_pcc', 'higher', 'gene variation across cells', gene=True, recommended=True, note='Requires paired target cells. Per-gene values require matrices, not aggregate logs.')
add('train_log_day_pcc_gene','higher','gene variation across cells',gene=True,eligible=False)
for axis in ['cell','gene']:
    add(f'train_log_mean_day_pcc_{axis}_unique', 'higher', 'unique-target '+axis+' PCC', gene=axis=='gene', recommended=True)
    add(f'train_log_day_pcc_{axis}_unique','higher','unique-target '+axis+' PCC',gene=axis=='gene',eligible=False)
    add(f'train_log_lower_day_pcc_{axis}','higher','conservative weighting diagnostic',eligible=False,note='Minimum of occurrence and unique averages; not a single consistent evaluation population.')
add('mse rmse mae','lower','absolute expression error',gene=True,recommended=True,note='Requires matched cells, genes and identical expression scale. Equal-day mean RMSE differs from pooled RMSE.')
add('r2','higher','explained squared error',gene=True,note='Global R2 uses a grand mean over all cell-gene entries; per-gene R2 has its own gene mean.')
add('mean_row_cosine','higher','uncentered cell profile',note='Requires pairing; zero norm undefined, retained as missing.')
add('pseudobulk_pcc pseudobulk_spearman mean_day_pseudobulk_pcc time_gene_pseudobulk_pcc','higher','population mean expression',recommended=True,note='Correlation across genes has no per-gene equivalent. Gene temporal PCC is a separate metric.')
add('pseudobulk_rmse pseudobulk_mae','lower','population mean error',gene='components',recommended=True,note='Per-gene signed/absolute mean differences are components, not a cross-gene correlation.')
add('pseudobulk_cosine','higher','uncentered population mean',gene=False)
add('variance_pcc','higher','gene variability',recommended=True,note='Across-gene variance correlation; no single-gene correlation equivalent.')
add('variance_rmse','lower','gene variability error',gene='components',recommended=True)
add('covariance_relative_frobenius','lower','gene co-variation',gene='incident-edge diagnostic',recommended=True,note='Population sample covariance. Single-gene covariance is variance, a distinct metric.')
add('covariance_edge_pcc','higher','gene co-variation',note='Correlation across gene-pair covariance edges; not individual gene PCC.')
add('mmd_u','lower','joint expression distribution',gene='different univariate metric',recommended=True,note='Unbiased-form MMD squared estimate can be negative. PCA, kernels and sample limits are part of the definition.')
add('sliced_wasserstein','lower','joint expression distribution',gene='different univariate metric',recommended=True,note='Truth-PCA projected multivariate distance; not a sum of per-gene distances.')
add('variogram','lower','gene-pair distribution',gene='incident-pair diagnostic',recommended=True,note='Mean squared difference of E sqrt(abs(gene_i-gene_j)); sampled pairs.')
add('de_score de_direction signed_overlap','higher','developmental change',gene='components',recommended=True,note='Needs verified observed earlier reference. Significant truth DE genes are gene diagnostics, not independent model replicates.')
add('chance_overlap n_true_up n_true_down','','truth/reference diagnostic',gene='diagnostic',eligible=False)
add('de_score_baseline de_score_upper de_direction_baseline de_direction_upper mmd_u_baseline mmd_u_upper variogram_baseline variogram_upper','','reference anchor diagnostic',eligible=False)
add('de_score_skill de_direction_skill mmd_u_skill variogram_skill virtual_embryo_t1_score','','legacy anchored diagnostic',eligible=False,note='Not monotone beyond empirical upper; keep separately and do not select as a core benchmark endpoint.')
add('legacy_ranked_signed_overlap legacy_ranked_de_score','','legacy rank-only DE diagnostic',eligible=False,note='Original source did not enforce sign on selected up/down sets.')
add('signed_bias','','calibration diagnostic',gene=True,eligible=False,note='Signed error: neither higher nor lower is uniformly better.')
add('detection_fraction_rmse detection_fraction_mae','lower','saved positive-value fraction',gene='components',note='Fraction of values >0 on the saved scale; not RNA detection unless that scale preserves zeros.')
add('gene_pcc gene_trajectory_pcc gene_cosine','higher','gene diagnostic',gene=True,eligible=False)
add('gene_temporal_rmse','lower','gene temporal mean error',gene=True,eligible=False)
add('gene_pcc_contribution gene_variance_error detection_fraction_error','','decomposition diagnostic',gene=True,eligible=False)
add('gene_wasserstein_1d gene_mmd_u_1d','lower','single-gene distribution diagnostic',gene=True,eligible=False,note='Distinct one-dimensional distances on a shared sample of up to 256 cells; not the joint truth-PCA metric. U-form MMD may be negative/undefined.')
add('gene_covariance_row_squared_error gene_covariance_relative_frobenius_component gene_variogram_incident_error','lower','incident gene-pair diagnostic',gene=True,eligible=False,note='Components of sampled multi-gene covariance/pair errors, not independent global endpoint scores.')
add('gene_covariance_profile_pcc','higher','gene covariance profile diagnostic',gene=True,eligible=False,note='Pearson across incident off-diagonal covariance edges; distinct from global edge Pearson.')
add('train_test_mse','lower','absolute expression error',day=True,gene=True,recommended=True,note='Explicit held-out test MSE from full training summary; validation loss is never substituted. Day/gene decomposition needs the original matched predictions, not a scalar summary.')
add('train_log_conservative_lower_pcc_cell train_log_conservative_lower_pcc_gene','','conservative weighting diagnostic',eligible=False,note='Minimum of two differently weighted averages; diagnostic only.')

def get_metric(name):
    return dict(METRICS.get(name, dict(metric=name,direction='',aspect='unregistered diagnostic',per_day=False,per_gene=False,recommended=False,benchmark_eligible=False,note='No definition inferred from a numeric column.')))

def display_model(model):
    if model in {'v6_200_new','v6_author'}: return 'unidirectional BLOOM'
    base = model.removesuffix('_200').removesuffix('_hvg').removesuffix('_author')
    return base.replace('Nonstationary_Transformer','Nonstationary Transformer')

def canonical_identity(dataset, panel, model, family=None):
    panel = 'hvg' if str(panel).lower() in {'200','hvg','hvg200'} else str(panel).lower()
    if dataset not in {'big','small','sted'} or panel not in {'hvg','author'}: raise ValueError('Explicit dataset big/small/sted and panel hvg/author required')
    if '/' in model:
        parts=model.split('/')
        if len(parts)!=3 or parts[0]!=dataset: raise ValueError('Slot dataset disagrees')
        family=family or parts[1]; model=parts[2]
    if model in {'v6_200_new','v6_author'}:
        expected='hvg' if model=='v6_200_new' else 'author'
        if panel!=expected: raise ValueError('FM model and native panel disagree')
        if family and family.upper()!='FM': raise ValueError('FM family conflicts')
        return dataset,panel,model,'FM'
    suffix='_200' if panel=='hvg' else '_author'
    base=model.removesuffix(suffix)
    if base in TRANS:
        if family and family.upper() not in {'TRANS','TRANSFORMER'}: raise ValueError('TRANS family conflicts')
        return dataset,panel,base+suffix,'TRANS'
    raise ValueError(f'Excluded or unknown benchmark model: {model}; only v6_200_new/v6_author and 22 TRANS are eligible')
