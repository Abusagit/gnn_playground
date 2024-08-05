def main(in1, in2, in3, mr_tables, token1=None, token2=None, param1=None, param2=None, html_file=None):
    import yt.wrapper as yt
    
    mr_table = mr_tables[0]
    
    yt.config.config['token'] = token1
    yt.config.config['proxy']['url'] = mr_table['cluster']
    
    # table has one row and 2 columns - pearson correlation with previous predictions and a tag
    run_results_dict = list(yt.read_table(mr_table['table'], format='json'))[0]
    main_metric = "pearson_correlation"

    metrics_list = []
    
    for metric_name, metric_value in run_results_dict.items():
        
        metric_dict = dict(
            name=metric_name,
            value=metric_value,
            main=metric_name == main_metric,
        )
            
        metrics_list.append(metric_dict)
        
    return metrics_list
