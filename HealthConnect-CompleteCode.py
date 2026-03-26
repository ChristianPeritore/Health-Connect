import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

np.random.seed(7)

# ---------------------------------------------------------
#  1. DATASET — 300 sessioni di allenamento da smartwatch
# ---------------------------------------------------------

n = 300

battiti   = np.random.normal(145, 20, n).clip(60, 200)    # bpm
sonno     = np.random.normal(7.0, 1.2, n).clip(3, 10)     # ore di sonno
carico    = np.random.normal(300, 80, n).clip(50, 600)     # minuti allenamento/settimana
riposo    = np.random.randint(0, 7, n)                     # giorni dall'ultimo riposo

# Livello rischio infortunio
punteggio = ((battiti > 170).astype(int) * 2 +
             (sonno < 6).astype(int) * 2 +
             (carico > 400).astype(int) * 1 +
             (riposo == 0).astype(int) * 1)

def a_classe(p):
    if p <= 1:   return 'BASSO'
    elif p <= 3: return 'MEDIO'
    else:        return 'ALTO'

rischio = np.array([a_classe(p) for p in punteggio])

df = pd.DataFrame({'battiti': battiti, 'sonno': sonno,
                   'carico': carico, 'riposo': riposo, 'rischio': rischio})

print("HEALTHCONNECT — Dataset:", len(df), "sessioni")
for c, n_ in df['rischio'].value_counts().items():
    print(f"  {c}: {n_}")
print(df.head(4).to_string(index=False))

# ---------------------------------------------------------
#  2. GRAFICO EDA — Medie variabili per livello di rischio
# ---------------------------------------------------------

ordine  = ['BASSO', 'MEDIO', 'ALTO']
colori  = ['#2ecc71', '#f39c12', '#e74c3c']

medie = df.groupby('rischio')[['battiti', 'sonno', 'carico']].mean().loc[ordine]
variabili = ['Battiti (bpm)', 'Ore sonno', 'Carico sett. (min)']

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(variabili))
larghezza = 0.25

for i, (classe, colore) in enumerate(zip(ordine, colori)):
    valori = medie.loc[classe].values
    # Normalizziamo per rendere confrontabili le scale diverse
    valori_norm = valori / np.array([200, 10, 600]) * 100
    barre = ax.bar(x + i * larghezza, valori_norm, larghezza,
                   label=classe, color=colore, edgecolor='black', linewidth=0.7)
    for b, v, vr in zip(barre, valori_norm, valori):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5,
                f'{vr:.0f}', ha='center', fontsize=8, fontweight='bold')

ax.set_title('HEALTHCONNECT — Medie Biometriche per Livello di Rischio', fontweight='bold')
ax.set_xticks(x + larghezza)
ax.set_xticklabels(variabili)
ax.set_ylabel('Valore normalizzato (% del massimo)')
ax.legend()
plt.tight_layout()
plt.savefig('healthconnect_grafico.png', dpi=120, bbox_inches='tight')
plt.show()
print("Grafico salvato: healthconnect_grafico.png")

# ---------------------------------------------------------
#  3. MODELLO ML
# ---------------------------------------------------------

le = LabelEncoder()
le.fit(ordine)
y = le.transform(df['rischio'])
X = df[['battiti', 'sonno', 'carico', 'riposo']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modello = RandomForestClassifier(n_estimators=50, random_state=42)
modello.fit(X_train, y_train)
y_pred = modello.predict(X_test)

print(f"\nAccuratezza del modello: {accuracy_score(y_test, y_pred)*100:.1f}%")

# ---------------------------------------------------------
#  4. GRAFICO RISULTATI — Matrice di confusione
# ---------------------------------------------------------

cm  = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap='YlOrRd')
ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
ax.set_xticklabels(ordine); ax.set_yticklabels(ordine)
ax.set_xlabel('Predizione'); ax.set_ylabel('Reale')
ax.set_title('HEALTHCONNECT — Matrice di Confusione', fontweight='bold')
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=16, fontweight='bold',
                color='white' if cm[i, j] > cm.max() * 0.6 else 'black')
plt.tight_layout()
plt.savefig('healthconnect_risultati.png', dpi=120, bbox_inches='tight')
plt.show()
print("Risultati salvati: healthconnect_risultati.png")

# ---------------------------------------------------------
#  5. SIMULATORE — Valutazione atleti
# ---------------------------------------------------------

consigli = {
    'ALTO':  'Riposo obbligatorio.',
    'MEDIO': 'Riduci intensita del 30%.',
    'BASSO': 'Condizione ottimale, allenati!'
}

print("\nSIMULATORE:")
atleti = pd.DataFrame({'battiti': [178, 130, 165],
                        'sonno':   [4.5, 8.0, 5.5],
                        'carico':  [480, 200, 380],
                        'riposo':  [0,   2,   0]})
nomi = ['Marco', 'Sofia', 'Luca']
pred = le.inverse_transform(modello.predict(atleti))
for nome, r in zip(nomi, pred):
    print(f"  {nome}: {r}  →  {consigli[r]}")