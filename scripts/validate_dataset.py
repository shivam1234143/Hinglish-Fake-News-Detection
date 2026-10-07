from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'data/hinglish_news.csv'
if not p.exists(): raise SystemExit(f'Missing {p}')
df=pd.read_csv(p)
req=['title','text','label']; miss=[c for c in req if c not in df.columns]
if miss: raise SystemExit(f'Missing columns: {miss}')
print('Rows:',len(df)); print('\nLabels:'); print(df.label.value_counts(dropna=False).sort_index())
print('\nMissing:'); print(df[req].isna().sum())
content=df.title.fillna('')+' '+df.text.fillna('')
print('\nExact duplicate content:',content.duplicated().sum())
print('Short records (<40 chars):',(content.str.len()<40).sum())
if 'source' in df: print('\nTop sources:'); print(df.source.value_counts().head(15))
if 'topic' in df: print('\nTopics:'); print(df.topic.value_counts().head(20))
if 'group_id' in df: print('\nGroups:',df.group_id.nunique())
