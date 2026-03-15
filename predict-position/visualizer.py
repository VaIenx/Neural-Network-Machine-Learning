import matplotlib
matplotlib.use('Agg')  # Speichert Plots als Dateien – funktioniert immer in VS Code

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os


class Visualizer:
    """
    Debugging & Übersicht für die F1-Punkte-Daten.
    Alle Plots helfen zu verstehen ob die Daten für das NN geeignet sind.

    Nutzung:
        viz = Visualizer(df)
        viz.plot_all()          # alle Plots auf einmal
        viz.gridpos_vs_points() # einzelne Plots

    Speichern:
        viz = Visualizer(df, save_dir="plots")   # Plots als PNG speichern
        viz.plot_all()
    """

    TEAM_COLORS = {
        'Mercedes':        '#00D2BE',
        'Red Bull Racing': '#0600EF',
        'Ferrari':         '#DC0000',
        'McLaren':         '#FF8700',
        'Alpine':          '#0090FF',
        'AlphaTauri':      '#2B4562',
        'Aston Martin':    '#006F62',
        'Williams':        '#005AFF',
        'Alfa Romeo':      '#900000',
        'Haas F1 Team':    '#FFFFFF',
    }

    def __init__(self, dataframe: pd.DataFrame, save_dir: str = None):
        """
        Parameters
        ----------
        dataframe : pd.DataFrame
            Der aufbereitete F1-Datensatz (aus DATA.csv).
        save_dir : str, optional
            Ordnerpfad in dem Plots gespeichert werden sollen.
            Standard: None  →  Plots werden nur angezeigt, nicht gespeichert.
        """
        self.df = dataframe.copy()
        self.save_dir = save_dir

        # Zielvariable berechnen: in den Punkten = Position 1-10
        if 'in_points' not in self.df.columns:
            self.df['in_points'] = (self.df['Position'] <= 10).astype(int)

        self._validate()

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"Plots werden gespeichert in: {os.path.abspath(self.save_dir)}/")

    def _validate(self):
        required = ['Abbreviation', 'TeamName', 'GridPosition', 'Position', 'in_points']
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Fehlende Spalten im DataFrame: {missing}")

    def _save_or_show(self, filename: str):
        if self.save_dir:
            path = os.path.join(self.save_dir, filename)
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Gespeichert: {path}")
            plt.close()
        else:
            plt.show()

    def _team_color(self, team: str) -> str:
        return self.TEAM_COLORS.get(team, '#888888')

    # ─────────────────────────────────────────────
    #  ÖFFENTLICHE METHODEN
    # ─────────────────────────────────────────────

    def plot_all(self):
        """Alle Plots nacheinander anzeigen oder speichern."""
        print("Erstelle alle Plots...\n")
        self.gridpos_vs_points()
        self.points_rate_per_team()
        self.quali_vs_points()
        self.class_balance()
        self.points_rate_per_gridpos()
        self.rainfall_effect()
        self.data_overview()
        print("\nFertig.")

    def gridpos_vs_points(self):
        """
        Kernplot: Startplatz vs. Wahrscheinlichkeit in den Punkten.
        Erwartung: Niedrige Startposition → hohe Punktewahrscheinlichkeit.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle("Startplatz vs. Punkte-Rate", fontsize=14, fontweight='bold')

        rate = self.df.groupby('GridPosition')['in_points'].mean()
        counts = self.df.groupby('GridPosition')['in_points'].count()

        bars = ax.bar(rate.index, rate.values * 100,
                      color='steelblue', edgecolor='black', linewidth=0.5, alpha=0.85)

        # Anzahl Fahrer als Annotation
        for pos, (r, c) in enumerate(zip(rate.values, counts.values)):
            ax.text(rate.index[pos], r * 100 + 1, f"n={c}",
                    ha='center', va='bottom', fontsize=7, color='gray')

        ax.axhline(50, color='red', linestyle='--', linewidth=1, label='50%')
        ax.set_xlabel("Startplatz (GridPosition)")
        ax.set_ylabel("In Punkten (%)")
        ax.set_ylim(0, 110)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        self._save_or_show("01_gridpos_vs_points.png")

    def points_rate_per_team(self):
        """
        Wie oft landet jedes Team in den Punkten?
        Zeigt ob TeamName ein nützliches Feature ist.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle("Punkte-Rate pro Team", fontsize=14, fontweight='bold')

        rate = (self.df.groupby('TeamName')['in_points']
                .mean()
                .sort_values(ascending=False))

        colors = [self._team_color(t) for t in rate.index]
        bars = ax.bar(rate.index, rate.values * 100,
                      color=colors, edgecolor='black', linewidth=0.5)

        for bar, val in zip(bars, rate.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val*100:.0f}%",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xlabel("Team")
        ax.set_ylabel("In Punkten (%)")
        ax.set_ylim(0, 110)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        self._save_or_show("02_points_rate_per_team.png")

    def quali_vs_points(self):
        """
        Qualizeit vs. Punkte-Wahrscheinlichkeit.
        Zeigt ob Q_best_sec ein gutes Feature ist.
        """
        if 'Q_best_sec' not in self.df.columns:
            print("Spalte Q_best_sec nicht vorhanden – Plot übersprungen.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle("Qualizeit vs. Punkte-Rate", fontsize=14, fontweight='bold')

        # Scatter: Qualizeit vs in_points (jitter für Lesbarkeit)
        colors = self.df['in_points'].map({1: '#2ecc71', 0: '#e74c3c'})
        jitter = np.random.uniform(-0.05, 0.05, len(self.df))
        ax.scatter(self.df['Q_best_sec'], self.df['in_points'] + jitter,
                   c=colors, alpha=0.3, s=10, edgecolors='none')

        # Binned mean
        bins = pd.cut(self.df['Q_best_sec'], bins=20)
        binned = self.df.groupby(bins, observed=True)['in_points'].mean()
        bin_centers = [interval.mid for interval in binned.index]
        ax.plot(bin_centers, binned.values, color='black',
                linewidth=2, label='Punkte-Rate (gebinnt)')

        ax.set_xlabel("Qualizeit Q_best_sec (Sekunden)")
        ax.set_ylabel("In Punkten (0/1)")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Nein', 'Ja'])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Legende manuell
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=8, label='In Punkten'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Nicht in Punkten'),
        ]
        ax.legend(handles=legend_elements + [ax.get_lines()[0]])

        plt.tight_layout()
        self._save_or_show("03_quali_vs_points.png")

    def class_balance(self):
        """
        Klassenverteilung: In Punkten vs. nicht.
        Wichtig für das NN – bei starkem Ungleichgewicht braucht man class_weight.
        """
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle("Klassenverteilung (Zielvariable)", fontsize=14, fontweight='bold')

        counts = self.df['in_points'].value_counts().sort_index()
        labels = ['Nicht in Punkten', 'In Punkten']
        colors = ['#e74c3c', '#2ecc71']

        # Pie
        axes[0].pie(counts.values, labels=labels, colors=colors,
                    autopct='%1.1f%%', startangle=90,
                    wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
        axes[0].set_title("Gesamt")

        # Pro Jahr (falls vorhanden)
        if 'year' in self.df.columns:
            pivot = self.df.groupby(['year', 'in_points']).size().unstack(fill_value=0)
            pivot.columns = labels
            pivot.plot(kind='bar', ax=axes[1], color=colors,
                       edgecolor='black', linewidth=0.5)
            axes[1].set_title("Pro Jahr")
            axes[1].set_xlabel("Jahr")
            axes[1].set_ylabel("Anzahl Fahrer")
            axes[1].tick_params(axis='x', rotation=0)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')
        else:
            axes[1].bar(labels, counts.values, color=colors,
                        edgecolor='black', linewidth=0.5)
            axes[1].set_ylabel("Anzahl")
            axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        self._save_or_show("04_class_balance.png")

    def points_rate_per_gridpos(self):
        """
        Heatmap: GridPosition vs. Team → Punkte-Rate.
        Zeigt Interaktionen zwischen den wichtigsten Features.
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.suptitle("Punkte-Rate: Startplatz × Team", fontsize=14, fontweight='bold')

        # Nur Top-Startplätze für Lesbarkeit
        top_grid = sorted(self.df['GridPosition'].dropna().unique())[:20]
        df_filtered = self.df[self.df['GridPosition'].isin(top_grid)]

        pivot = df_filtered.pivot_table(
            values='in_points', index='TeamName',
            columns='GridPosition', aggfunc='mean'
        )

        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Punkte-Rate')

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([int(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Startplatz")
        ax.set_ylabel("Team")

        # Werte in Zellen
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0%}", ha='center', va='center',
                            fontsize=7, color='black' if 0.3 < val < 0.7 else 'white')

        plt.tight_layout()
        self._save_or_show("05_heatmap_team_gridpos.png")

    def rainfall_effect(self):
        """
        Effekt von Regen auf die Punkte-Rate pro Startplatz.
        Nur sichtbar wenn 'rainfall' Spalte vorhanden ist.
        """
        if 'rainfall' not in self.df.columns:
            print("Spalte 'rainfall' nicht vorhanden – Plot übersprungen.")
            return

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle("Regen-Effekt: Startplatz vs. Punkte-Rate", fontsize=14, fontweight='bold')

        for rain, label, color in [(False, 'Trocken', '#3498db'), (True, 'Regen', '#95a5a6')]:
            subset = self.df[self.df['rainfall'] == rain]
            rate = subset.groupby('GridPosition')['in_points'].mean()
            ax.plot(rate.index, rate.values * 100,
                    label=label, color=color, linewidth=2,
                    marker='o', markersize=4)

        ax.axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel("Startplatz")
        ax.set_ylabel("In Punkten (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_or_show("06_rainfall_effect.png")

    def data_overview(self):
        """
        Übersichts-Dashboard: Datenmenge, Verteilungen, Kennzahlen.
        """
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("Daten-Übersicht (Debug)", fontsize=14, fontweight='bold')

        # Oben links: GridPosition Histogramm
        axes[0, 0].hist(self.df['GridPosition'].dropna(), bins=20,
                        color='steelblue', edgecolor='black', linewidth=0.3)
        axes[0, 0].set_title("GridPosition Verteilung")
        axes[0, 0].set_xlabel("Startplatz")
        axes[0, 0].set_ylabel("Anzahl Fahrer")
        axes[0, 0].grid(True, alpha=0.3)

        # Oben rechts: Q_best_sec Histogramm
        if 'Q_best_sec' in self.df.columns:
            axes[0, 1].hist(self.df['Q_best_sec'].dropna(), bins=40,
                            color='darkorange', edgecolor='black', linewidth=0.3)
            axes[0, 1].set_title("Q_best_sec Verteilung")
            axes[0, 1].set_xlabel("Qualizeit (Sekunden)")
        else:
            axes[0, 1].text(0.5, 0.5, 'Q_best_sec\nnicht vorhanden',
                            ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].grid(True, alpha=0.3)

        # Unten links: Datenmenge pro Jahr
        if 'year' in self.df.columns:
            year_counts = self.df['year'].value_counts().sort_index()
            axes[1, 0].bar(year_counts.index.astype(str), year_counts.values,
                           color='mediumseagreen', edgecolor='black', linewidth=0.5)
            axes[1, 0].set_title("Datenmenge pro Jahr")
            axes[1, 0].set_xlabel("Jahr")
            axes[1, 0].set_ylabel("Anzahl Einträge")
            for i, (yr, cnt) in enumerate(year_counts.items()):
                axes[1, 0].text(i, cnt + 2, str(cnt),
                                ha='center', fontsize=9, fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'Keine year-Spalte',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Unten rechts: Kennzahlen-Tabelle
        axes[1, 1].axis('off')
        in_pts = self.df['in_points'].sum()
        total = len(self.df)
        stats = {
            "Gesamt Einträge":        total,
            "In Punkten":             int(in_pts),
            "Nicht in Punkten":       int(total - in_pts),
            "Klassen-Ratio":          f"{in_pts/total*100:.1f}% / {(1-in_pts/total)*100:.1f}%",
            "Rennen":                 self.df['race'].nunique() if 'race' in self.df.columns else '–',
            "Teams":                  self.df['TeamName'].nunique(),
            "Fahrer":                 self.df['Abbreviation'].nunique(),
            "Jahre":                  str(sorted(self.df['year'].unique().tolist())) if 'year' in self.df.columns else '–',
            "Fehlende Werte":         int(self.df.isnull().sum().sum()),
        }
        rows = [[k, v] for k, v in stats.items()]
        table = axes[1, 1].table(cellText=rows, colLabels=["Kennzahl", "Wert"],
                                  loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.6)
        axes[1, 1].set_title("Zusammenfassung", fontweight='bold', pad=12)

        plt.tight_layout()
        self._save_or_show("07_data_overview.png")


if __name__ == "__main__":
    df = pd.read_csv('DATA.csv')
    viz = Visualizer(df, save_dir='predict-position/plots')
    viz.plot_all()