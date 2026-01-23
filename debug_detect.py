
import pandas as pd
import numpy as np
import os

SESSION_PATH = "/home/matthew/PycharmProjects/gsr_meter/Session_Data/Collect For"

def load_data(session_path):
    csv_path = os.path.join(session_path, "GSR.csv")
    df = pd.read_csv(csv_path)
    df['Seconds'] = df['Elapsed'].apply(lambda x: sum(float(c) * 60**i for i, c in enumerate(reversed(str(x).split(':')))))
    df['TA_Smooth'] = df['TA'].rolling(window=10, center=True).mean().fillna(df['TA'])
    return df

def debug_grouping():
    df = load_data(SESSION_PATH)
    # The current grouping logic:
    # df['Incident_Group'] = (df['Pattern'] != df['Pattern'].shift()).cumsum()
    
    # Let's try a logic that groups ANY sequence of patterns together if they are adjacent
    df['Is_Incident'] = df['Pattern'].notna() & (~df['Pattern'].isin(['MOTION', 'CALIB_START', 'SESSION_STARTED']))
    df['Group'] = (df['Is_Incident'] != df['Is_Incident'].shift()).cumsum()
    
    groups = df[df['Is_Incident']].groupby('Group')
    print(f"Total Combined Incident Groups: {len(groups)}")
    
    results = []
    for g_id, group in groups:
        start_idx = group.index[0]
        end_idx = group.index[-1]
        p_list = group['Pattern'].unique()
        
        # Simplified Peak-to-Trough check
        peak_idx = df.iloc[max(0, start_idx-20):min(len(df)-1, end_idx+10)]['TA_Smooth'].idxmax()
        trough_idx = df.iloc[peak_idx:min(len(df)-1, end_idx+20)]['TA_Smooth'].idxmin()
        ta_drop = df.loc[peak_idx, 'TA'] - df.loc[trough_idx, 'TA']
        
        results.append({
            'start': df.loc[start_idx, 'Seconds'],
            'ta_drop': ta_drop,
            'patterns': p_list
        })
    
    results.sort(key=lambda x: x['ta_drop'], reverse=True)
    
    print("\nTop 20 Combined Drops by TA:")
    for r in results[:20]:
        print(f"Time: {r['start']:.1f}s | TA Drop: {r['ta_drop']:.4f} | Patterns: {r['patterns']}")

if __name__ == "__main__":
    debug_grouping()
