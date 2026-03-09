import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os


class Visualizer:
    """
    Debugging & Übersicht für die F1-Daten.
    Alle Plots helfen zu verstehen ob die Daten für das NN geeignet sind.

    Nutzung:
        viz = Visualizer(df)
        viz.plot_all()          # alle Plots auf einmal
        viz.degradation()       # einzelne Plots

    Speichern:
        viz = Visualizer(df, save_dir="plots")   # Plots als PNG speichern
        viz.plot_all()
    """

    COMPOUND_COLORS = {
        'SOFT':   '#E8002D',  # Rot  – wie Pirelli
        'MEDIUM': '#FFF200',  # Gelb – wie Pirelli
        'HARD':   '#EBEBEB',  # Weiß – wie Pirelli
    }

    def __init__(self, dataframe: pd.DataFrame, save_dir: str):
        """
        Parameters
        ----------
        dataframe : pd.DataFrame
            Der aufbereitete F1-Datensatz (aus DATA.csv).
        save_dir : str, optional
            Ordnerpfad in dem Plots gespeichert werden sollen.
            Beispiel: save_dir="plots"  →  speichert in ./plots/
            Standard: None  →  Plots werden nur angezeigt, nicht gespeichert.
        """
        self.df = dataframe.copy()
        self.save_dir = save_dir
        self._validate()

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"Plots werden gespeichert in: {os.path.abspath(self.save_dir)}/")

    def _validate(self):
        required = ['LapTime', 'LapNumber', 'TyreLife', 'Compound', 'Team', 'GP']
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Fehlende Spalten im DataFrame: {missing}")

    def _save_or_show(self, filename: str):
        """Speichert den aktuellen Plot oder zeigt ihn an – je nach Konfiguration."""
        if self.save_dir:
            path = os.path.join(self.save_dir, filename)
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Gespeichert: {path}")
            plt.close()
        else:
            plt.show()

    # ─────────────────────────────────────────────
    #  ÖFFENTLICHE METHODEN
    # ─────────────────────────────────────────────

    def plot_all(self):
        """Alle Plots nacheinander anzeigen oder speichern."""
        print("Erstelle alle Plots...\n")
        self.degradation()
        self.laptimes_per_gp()
        self.compound_distribution()
        self.tyrelife_vs_laptime()
        self.lapnumber_vs_laptime()
        self.team_comparison()
        self.data_overview()
        print("\nFertig.")

    def degradation(self):
        """
        Kernplot: TyreLife vs LapTime pro Compound.
        Zeigt ob das Netz überhaupt ein Signal zum Lernen hat.
        Erwartung: Kurve steigt mit TyreLife an (Reifen wird langsamer).
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        fig.suptitle("Reifendegradation: TyreLife vs LapTime", fontsize=14, fontweight='bold')

        compounds = ['SOFT', 'MEDIUM', 'HARD']

        for ax, compound in zip(axes, compounds):
            data = self.df[self.df['Compound'] == compound]
            color = self.COMPOUND_COLORS.get(compound, 'gray')

            if len(data) == 0:
                ax.text(0.5, 0.5, 'Keine Daten', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(compound)
                continue

            ax.scatter(data['TyreLife'], data['LapTime'],
                       alpha=0.15, s=8, color=color, edgecolors='none')

            trend = data.groupby('TyreLife')['LapTime'].median()
            ax.plot(trend.index, trend.values,
                    color='black', linewidth=2, label='Median')

            ax.set_title(compound, fontweight='bold', color=color if compound != 'HARD' else '#333333')
            ax.set_xlabel("TyreLife (Runden)")
            ax.set_ylabel("LapTime (Sekunden)" if compound == 'SOFT' else "")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            ax.text(0.98, 0.02, f"n={len(data)}", transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8, color='gray')

        plt.tight_layout()
        self._save_or_show("01_degradation.png")

    def laptimes_per_gp(self):
        """
        Boxplot der LapTimes pro Grand Prix.
        Zeigt ob einzelne Rennen Ausreißer sind oder ob die Strecken
        stark unterschiedliche Grundgeschwindigkeiten haben.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle("LapTime-Verteilung pro Grand Prix", fontsize=14, fontweight='bold')

        gp_order = sorted(self.df['GP'].unique())
        data_per_gp = [self.df[self.df['GP'] == gp]['LapTime'].values for gp in gp_order]

        bp = ax.boxplot(data_per_gp, labels=gp_order, patch_artist=True)

        colors = plt.cm.Set2(np.linspace(0, 1, len(gp_order)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel("Grand Prix")
        ax.set_ylabel("LapTime (Sekunden)")
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=15)

        for i, gp in enumerate(gp_order):
            median = self.df[self.df['GP'] == gp]['LapTime'].median()
            ax.text(i + 1, median + 0.3, f"{median:.1f}s",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

        plt.tight_layout()
        self._save_or_show("02_laptimes_per_gp.png")

    def compound_distribution(self):
        """
        Wie viele Runden gibt es pro Compound & GP?
        Zeigt ob ein Klassenungleichgewicht vorliegt.
        """
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Compound-Verteilung", fontsize=14, fontweight='bold')

        counts = self.df['Compound'].value_counts()
        colors = [self.COMPOUND_COLORS.get(c, 'gray') for c in counts.index]
        axes[0].pie(counts.values, labels=counts.index, colors=colors,
                    autopct='%1.1f%%', startangle=90,
                    wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
        axes[0].set_title("Gesamt")

        pivot = self.df.groupby(['GP', 'Compound']).size().unstack(fill_value=0)
        compound_order = [c for c in ['SOFT', 'MEDIUM', 'HARD'] if c in pivot.columns]
        pivot = pivot[compound_order]
        bar_colors = [self.COMPOUND_COLORS.get(c, 'gray') for c in compound_order]

        pivot.plot(kind='bar', ax=axes[1], color=bar_colors,
                   edgecolor='black', linewidth=0.5)
        axes[1].set_title("Pro Grand Prix")
        axes[1].set_xlabel("Grand Prix")
        axes[1].set_ylabel("Anzahl Runden")
        axes[1].tick_params(axis='x', rotation=15)
        axes[1].legend(title="Compound")
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        self._save_or_show("03_compound_distribution.png")

    def tyrelife_vs_laptime(self):
        """
        TyreLife vs LapTime – alle Teams übereinander.
        Zeigt ob Teams unterschiedliche Degradationskurven haben.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle("TyreLife vs LapTime pro Team (alle Compounds)", fontsize=14, fontweight='bold')

        teams = self.df['Team'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(teams)))

        for team, color in zip(teams, colors):
            data = self.df[self.df['Team'] == team]
            trend = data.groupby('TyreLife')['LapTime'].median()
            ax.plot(trend.index, trend.values,
                    label=team, color=color, linewidth=2, marker='o', markersize=3)

        ax.set_xlabel("TyreLife (Runden)")
        ax.set_ylabel("Median LapTime (Sekunden)")
        ax.legend(title="Team", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_or_show("04_tyrelife_vs_laptime_by_team.png")

    def lapnumber_vs_laptime(self):
        """
        LapNumber vs LapTime – zeigt den Tankgewichtseffekt.
        Erwartung: Erste Runden langsamer (schweres Auto),
        dann schneller bis der Reifen degradiert.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle("LapNumber vs LapTime (Tankgewichtseffekt)", fontsize=14, fontweight='bold')

        for compound, color in self.COMPOUND_COLORS.items():
            data = self.df[self.df['Compound'] == compound]
            if len(data) == 0:
                continue
            trend = data.groupby('LapNumber')['LapTime'].median()

            if compound == 'HARD':
                ax.plot(trend.index, trend.values, label=compound, color=color,
                        linewidth=2, path_effects=[
                            __import__('matplotlib.patheffects', fromlist=['withStroke'])
                            .withStroke(linewidth=3, foreground='black')
                        ])
            else:
                ax.plot(trend.index, trend.values, label=compound,
                        color=color, linewidth=2)

        ax.set_xlabel("LapNumber (Runde im Rennen)")
        ax.set_ylabel("Median LapTime (Sekunden)")
        ax.legend(title="Compound")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_or_show("05_lapnumber_vs_laptime.png")

    def team_comparison(self):
        """
        Boxplot LapTime pro Team.
        Zeigt wie groß der Unterschied zwischen Teams ist.
        Wenn der Unterschied groß ist → Team als Feature sinnvoll.
        """
        fig, ax = plt.subplots(figsize=(11, 5))
        fig.suptitle("LapTime-Verteilung pro Team", fontsize=14, fontweight='bold')

        teams = sorted(self.df['Team'].unique())
        data_per_team = [self.df[self.df['Team'] == t]['LapTime'].values for t in teams]

        bp = ax.boxplot(data_per_team, labels=teams, patch_artist=True)
        colors = plt.cm.Set1(np.linspace(0, 1, len(teams)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel("Team")
        ax.set_ylabel("LapTime (Sekunden)")
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=10)

        plt.tight_layout()
        self._save_or_show("06_team_comparison.png")

    def data_overview(self):
        """
        Übersichts-Dashboard: Datenmenge, Werteverteilungen, Korrelationen.
        """
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("Daten-Übersicht (Debug)", fontsize=14, fontweight='bold')

        # Oben links: LapTime Histogramm
        axes[0, 0].hist(self.df['LapTime'], bins=50, color='steelblue', edgecolor='black', linewidth=0.3)
        axes[0, 0].set_title("LapTime Verteilung")
        axes[0, 0].set_xlabel("Sekunden")
        axes[0, 0].set_ylabel("Anzahl Runden")
        axes[0, 0].axvline(self.df['LapTime'].mean(), color='red', linestyle='--',
                           label=f"Mean: {self.df['LapTime'].mean():.1f}s")
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)

        # Oben rechts: TyreLife Histogramm
        axes[0, 1].hist(self.df['TyreLife'], bins=40, color='darkorange', edgecolor='black', linewidth=0.3)
        axes[0, 1].set_title("TyreLife Verteilung")
        axes[0, 1].set_xlabel("Runden auf dem Reifen")
        axes[0, 1].set_ylabel("Anzahl")
        axes[0, 1].grid(True, alpha=0.3)

        # Unten links: Datenmenge pro GP
        gp_counts = self.df['GP'].value_counts()
        axes[1, 0].bar(gp_counts.index, gp_counts.values,
                       color='mediumseagreen', edgecolor='black', linewidth=0.5)
        axes[1, 0].set_title("Datenmenge pro Grand Prix")
        axes[1, 0].set_xlabel("Grand Prix")
        axes[1, 0].set_ylabel("Anzahl Runden")
        axes[1, 0].tick_params(axis='x', rotation=15)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        for i, (gp, count) in enumerate(gp_counts.items()):
            axes[1, 0].text(i, count + 5, str(count), ha='center', fontsize=9, fontweight='bold')

        # Unten rechts: Kennzahlen-Tabelle
        axes[1, 1].axis('off')
        stats = {
            "Gesamt Runden":      len(self.df),
            "Rennen":             self.df['GP'].nunique(),
            "Teams":              self.df['Team'].nunique(),
            "Compounds":          self.df['Compound'].nunique(),
            "LapTime Min":        f"{self.df['LapTime'].min():.2f}s",
            "LapTime Max":        f"{self.df['LapTime'].max():.2f}s",
            "LapTime Mittelwert": f"{self.df['LapTime'].mean():.2f}s",
            "Max TyreLife":       int(self.df['TyreLife'].max()),
            "Fehlende Werte":     int(self.df.isnull().sum().sum()),
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