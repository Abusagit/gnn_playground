def main(in1, in2, in3, mr_tables, token1=None, token2=None, param1=None, param2=None, html_file=None):
    print('in1:', in1)
    print('in2:', in2)
    print('in3:', in3)
    print('mr_tables:', mr_tables)

    # read all data from first mr table
    import yt.wrapper as yt
    yt.config.config['token'] = token1
    yt.config.config['proxy']['url'] = mr_tables[0]['cluster']

    return {"mr_tables": mr_tables}