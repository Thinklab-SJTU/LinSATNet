def get_snp500_keys():
    snp500 = [
        'AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOGL', 'GOOG', 'BRK-B', 'UNH', 'NVDA', 'JNJ', 'FB', 'PG', 'JPM', 'XOM', 'V', 'HD',
        'CVX', 'MA', 'ABBV', 'PFE', 'BAC', 'KO', 'COST', 'AVGO', 'PEP', 'WMT', 'LLY', 'TMO', 'VZ', 'CSCO', 'DIS', 'MRK',
        'ABT', 'CMCSA', 'ACN', 'ADBE', 'INTC', 'MCD', 'WFC', 'CRM', 'DHR', 'BMY', 'NKE', 'TXN', 'PM', 'LIN', 'RTX', 'QCOM',
        'UNP', 'NEE', 'MDT', 'AMD', 'AMGN', 'T', 'UPS', 'SPGI', 'CVS', 'LOW', 'HON', 'INTU', 'COP', 'PLD', 'IBM', 'ANTM',
        'MS', 'ORCL', 'AMT', 'CAT', 'TGT', 'DE', 'AXP', 'GS', 'LMT', 'SCHW', 'C', 'MO', 'PYPL', 'AMAT', 'GE', 'BA', 'NFLX',
        'BLK', 'NOW', 'ADP', 'BKNG', 'MDLZ', 'ISRG', 'SBUX', 'CB', 'DUK', 'MMC', 'ZTS', 'MMM', 'CCI', 'SYK', 'CI', 'ADI',
        'SO', 'CME', 'GILD', 'MU', 'CSX', 'TMUS', 'TJX', 'EW', 'REGN', 'PNC', 'BDX', 'AON', 'D', 'USB', 'VRTX', 'CL', 'EOG',
        'TFC', 'EQIX', 'ICE', 'NOC', 'LRCX', 'PGR', 'BSX', 'NSC', 'EL', 'FCX', 'ATVI', 'PSA', 'CHTR', 'FIS', 'WM', 'F',
        'NEM', 'SHW', 'SLB', 'ITW', 'ETN', 'DG', 'FISV', 'GM', 'HUM', 'COF', 'EMR', 'GD', 'SRE', 'APD', 'PXD', 'MCO', 'ADM',
        'DOW', 'AEP', 'MPC', 'ILMN', 'HCA', 'AIG', 'FDX', 'OXY', 'MRNA', 'KLAC', 'CNC', 'MAR', 'MET', 'LHX', 'ROP', 'MCK',
        'ORLY', 'EXC', 'KMB', 'NXPI', 'WBD', 'SYY', 'AZO', 'JCI', 'NUE', 'GIS', 'SNPS', 'PRU', 'IQV', 'CTSH', 'ECL', 'HLT',
        'DLR', 'WMB', 'DXCM', 'CTVA', 'VLO', 'PAYX', 'TRV', 'O', 'APH', 'CMG', 'WELL', 'SPG', 'STZ', 'FTNT', 'ADSK', 'CDNS',
        'TEL', 'IDXX', 'SBAC', 'XEL', 'HPQ', 'PSX', 'TWTR', 'GPN', 'KR', 'AFL', 'MSI', 'DLTR', 'PEG', 'KMI', 'AJG', 'ALL',
        'MSCI', 'ROST', 'A', 'MCHP', 'DVN', 'BAX', 'EA', 'CTAS', 'CARR', 'PH', 'YUM', 'AVB', 'DD', 'TT', 'ED', 'HAL',
        'VRSK', 'RMD', 'EBAY', 'TDG', 'BK', 'WEC', 'FAST', 'WBA', 'HSY', 'DFS', 'MNST', 'IFF', 'SIVB', 'ES', 'PPG', 'AMP',
        'OTIS', 'WY', 'EQR', 'MTB', 'BIIB', 'TROW', 'OKE', 'KHC', 'ROK', 'AWK', 'PCAR', 'MTD', 'AME', 'HES', 'BKR', 'WTW',
        'APTV', 'CMI', 'ARE', 'EXR', 'CBRE', 'BLL', 'FRC', 'LYB', 'TSN', 'DAL', 'RSG', 'LUV', 'CERN', 'EXPE', 'EIX', 'KEYS',
        'DTE', 'ALGN', 'ANET', 'FE', 'FITB', 'ZBH', 'STT', 'WST', 'MKC', 'GLW', 'CHD', 'ODFL', 'LH', 'AEE', 'CPRT', 'EFX',
        'ETR', 'MOS', 'IT', 'ANSS', 'HIG', 'ABC', 'MAA', 'TSCO', 'CTRA', 'ALB', 'STE', 'VTR', 'DHI', 'CDW', 'SWK', 'DRE',
        'ESS', 'URI', 'PPL', 'VMC', 'ULTA', 'FANG', 'MLM', 'NTRS', 'MTCH', 'TDY', 'GWW', 'CF', 'CMS', 'CFG', 'FTV', 'ENPH',
        'DOV', 'HPE', 'FLT', 'CINF', 'RF', 'CEG', 'LEN', 'ZBRA', 'CNP', 'VRSN', 'HBAN', 'SYF', 'BBY', 'MRO', 'NDAQ', 'KEY',
        'PKI', 'COO', 'RJF', 'GPC', 'MOH', 'AKAM', 'SWKS', 'HOLX', 'PARA', 'IP', 'IR', 'PEAK', 'J', 'CLX', 'RCL', 'WAT',
        'AMCR', 'BXP', 'TER', 'VFC', 'PFG', 'K', 'MPWR', 'UDR', 'DRI', 'CPT', 'CAH', 'CAG', 'PWR', 'BR', 'FMC', 'NTAP',
        'EXPD', 'WAB', 'TRMB', 'POOL', 'GRMN', 'OMC', 'STX', 'UAL', 'IRM', 'DGX', 'SBNY', 'EVRG', 'CTLT', 'FDS', 'ATO',
        'TTWO', 'BRO', 'LNT', 'TYL', 'KIM', 'EPAM', 'TECH', 'CE', 'WDC', 'MGM', 'SJM', 'PKG', 'XYL', 'LDOS', 'HRL', 'CCL',
        'GNRC', 'TFX', 'AES', 'TXT', 'NLOK', 'APA', 'KMX', 'JKHY', 'HST', 'IEX', 'INCY', 'LYV', 'WRB', 'CZR', 'JBHT',
        'PAYC', 'BBWI', 'NVR', 'DPZ', 'AVY', 'IPG', 'EMN', 'CRL', 'AAP', 'HWM', 'CHRW', 'ABMD', 'LKQ', 'WRK', 'SEDG', 'AAL',
        'L', 'RHI', 'CTXS', 'LVS', 'ETSY', 'VTRS', 'FOXA', 'MAS', 'BF-B', 'HSIC', 'CBOE', 'QRVO', 'FFIV', 'NI', 'NDSN',
        'SNA', 'BIO', 'JNPR', 'RE', 'HAS', 'LNC', 'CMA', 'REG', 'AIZ', 'PHM', 'WHR', 'PTC', 'ALLE', 'TAP', 'MKTX', 'LUMN',
        'LW', 'SEE', 'UHS', 'FBHS', 'GL', 'NLSN', 'CPB', 'NRG', 'ZION', 'BWA', 'XRAY', 'HII', 'NCLH', 'PNW', 'PNR', 'TPR',
        'NWL', 'AOS', 'FRT', 'OGN', 'NWSA', 'WYNN', 'CDAY', 'DISH', 'ROL', 'DXC', 'BEN', 'ALK', 'IVZ', 'MHK', 'DVA', 'VNO',
        'PENN', 'PVH', 'RL', 'FOX', 'IPGP', 'UAA', 'UA', 'NWS', 'EMBC']
    snp500.remove('EMBC')
    snp500.remove('CEG')
    snp500.remove('CARR')
    snp500.remove('CDAY')
    snp500.remove('CTVA')
    snp500.remove('DOW')
    snp500.remove('FOX')
    snp500.remove('FOXA')
    snp500.remove('MRNA')
    snp500.remove('OGN')
    snp500.remove('OTIS')
    return snp500
