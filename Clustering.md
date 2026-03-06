**Hybrider Clustering-Ansatz aus Literaturrecherche**

Dieser Ansatz kombiniert die Extraktion von aussagekräftigen Finanzkennzahlen mit einem zeitreihenspezifischen Gruppierungsverfahren, um Finanzwerte (Assets) mit ähnlichem Marktverhalten in Cluster zusammenzufassen. Der Prozess gliedert sich in vier wesentliche Schritte:

1. *Datenaufbereitung und Feature-Extraktion*:
Anstatt lediglich die reinen, absoluten Preisverläufe der Assets zu betrachten, werden die Daten zunächst in normalisierte Zeitreihen und relevante Finanzmetriken (Features) transformiert. Hierbei werden für jedes Asset Eigenschaften wie tägliche Renditen, die rollierende Volatilität (Risiko), Drawdowns oder Momentum-Indikatoren berechnet. Dies stellt sicher, dass Assets nicht aufgrund ihrer unterschiedlichen Preisniveaus (z. B. ein Asset bei 60.000 € vs. eines bei 0,50 €), sondern aufgrund ihrer tatsächlichen Marktdynamik objektiv verglichen werden.

2. *Distanzmessung mittels Dynamic Time Warping (DTW)*:
Um zu berechnen, wie ähnlich sich die Verhaltensmuster zweier Assets sind, wird das Distanzmaß Dynamic Time Warping (DTW) verwendet.
Im Gegensatz zu klassischen Methoden, die Datenpunkte starr zum exakt selben Zeitpunkt vergleichen, kann DTW zeitliche Verzögerungen (Phasenverschiebungen) abgleichen. Reagiert beispielsweise ein Asset sofort auf ein Marktereignis und ein zweites Asset zeigt exakt dasselbe Kursmuster, jedoch erst mit einem Tag Verzögerung, erkennt DTW diese strukturelle Ähnlichkeit.

3. *Gruppierung durch K-Means Clustering*:
Basierend auf den mit DTW berechneten Distanzen wird der K-Means-Algorithmus angewendet.
Dieser Algorithmus teilt die Menge der Assets iterativ in eine bestimmte Anzahl von Gruppen (Clustern) ein. Das Ziel des Algorithmus ist es, die Zuweisungen so zu optimieren, dass die Assets innerhalb eines Clusters extrem ähnliche Verhaltensmuster aufweisen, während sich die Cluster untereinander so stark wie möglich unterscheiden. In Python lässt sich dies effizient über die Bibliothek tslearn umsetzen, welche K-Means direkt mit DTW verknüpft.

4. *Evaluierung und Bestimmung der Cluster-Anzahl*:
Da die optimale Anzahl der Cluster (k) für die 100 Assets im Voraus unbekannt ist, wird das Clustering-Verfahren für verschiedene Werte (z. B. k=3 bis k=15) mehrfach durchgeführt. Die Qualität der jeweiligen Gruppierungen wird anschließend durch den sogenannten Silhouette Score bewertet. Diese Metrik misst, wie gut jedes Asset zu seinem zugewiesenen Cluster im Vergleich zu den benachbarten Clustern passt. Der Durchlauf, der den höchsten Silhouette Score erzielt, definiert die mathematisch fundierteste und endgültige Cluster-Anzahl für die Konfiguration des Trading-Bots.
