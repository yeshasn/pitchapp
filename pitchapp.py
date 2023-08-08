from baseball_scraper import statcast
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
import streamlit as st

df = pd.read_csv("playoff1.csv")

narrow_df = df[['player_name', 'stand', 'balls', 'strikes', 'outs_when_up', 'on_1b', 'on_2b', 'on_3b', 'pitch_name']]
narrow_df.loc[:,'on_1b'] = narrow_df.loc[:,'on_1b'].fillna(0)
narrow_df.loc[:,'on_2b'] = narrow_df.loc[:,'on_2b'].fillna(0)
narrow_df.loc[:,'on_3b'] = narrow_df.loc[:,'on_3b'].fillna(0)

narrow_df.loc[:,'on_1b'] = narrow_df.loc[:,'on_1b'].mask(narrow_df.loc[:,'on_1b'] > 0, 1)
narrow_df.loc[:,'on_2b'] = narrow_df.loc[:,'on_2b'].mask(narrow_df.loc[:,'on_2b'] > 0, 1)
narrow_df.loc[:,'on_3b'] = narrow_df.loc[:,'on_3b'].mask(narrow_df.loc[:,'on_3b'] > 0, 1)

#narrow_df = narrow_df.sample(frac = 1)
narrow_df = narrow_df.dropna(subset= ['pitch_name'])

narrow_df.loc[:,'stand'] = narrow_df.loc[:,'stand'].replace('L', 0)
narrow_df.loc[:,'stand'] = narrow_df.loc[:,'stand'].replace('R', 1)

st.write("""
# MLB Pitch predictor
Enter pitcher name then use the sliders to predict a pitch
""")

st.markdown("---")

name = st.text_input("Enter pitcher name (Last, First)", placeholder = "Verlander, Justin")
if name is None:
    st.error("Please enter a valid name")
    st.stop()


st.sidebar.header("FAQ")
with st.sidebar.expander("Why am I getting an error?"):
    st.write("This is because you have not entered a valid pitcher's name. Try again with a valid pitcher's name.")
st.sidebar.markdown("---")

new_df = narrow_df[narrow_df['player_name'] == name]
del new_df['player_name']

length = len(new_df)
first = name[name.index(",")+1:]
last = name[:name.index(",")]
st.write(" ")
st.write(first, last, "has thrown", length, "pitches between April 7th and September 1st 2022")
new_df

X = []
y = []

for i in range(len(new_df)):
    X.append(list(new_df.iloc[i][:len(new_df.columns) - 1]))
    y.append(new_df.iloc[i][-1])

train_pct = 1

train_X = X[:int(train_pct*len(X))]
train_y = y[:int(train_pct*len(y))]

kn = KNeighborsClassifier()
kn.fit(train_X, train_y)
ad = AdaBoostClassifier()
ad.fit(train_X, train_y)
sv = SVC()
sv.fit(train_X, train_y)

stand = st.sidebar.number_input('stance (0 = lefty, 1 = righty)', 0, 1, 0)
balls = st.sidebar.number_input('balls', 0, 3, 0)
strikes = st.sidebar.number_input('strikes', 0, 2, 0)
outs = st.sidebar.number_input('outs', 0, 2, 0)
run_one = st.sidebar.number_input('runner on first (0 = no, 1 = yes)', 0, 1, 0)
run_two = st.sidebar.number_input('runner on second (0 = no, 1 = yes)', 0, 1, 0)
run_three = st.sidebar.number_input('runner on third (0 = no, 1 = yes)', 0, 1, 0)

values = [stand, balls, strikes, outs, run_one, run_two, run_three]
prediction1 = kn.predict([values])[0]
prediction2 = ad.predict([values])[0]
prediction3 = sv.predict([values])[0]
st.markdown("---")
st.write("__Predictions:__ ", prediction1, ", ", prediction2, ", ", prediction3)
