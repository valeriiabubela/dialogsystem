This project was done as a part of the seminar "Praktische Anwendungen in Berufsfeldern: Dialogsystem" at the Technical University of Berlin.
During the seminar, the fundamental modules of a dialog system (e.g. (speech) input, semantic parsing, data binding, (speech) output) were discussed, analyzed und built. As a result, this <b>Corona Chatbot</b> was created.
<ul>
<li>The bot handles 5 query types: <b>new cases, incidence, deaths, vaccinations and recovered</b> for Germany as well as German single states using 3 endpoints from the <a href="https://github.com/marlon360/rki-covid-api" target="_blank">RKI Covid API</a>.</li>
<li>The bot is trained to identify a query type using a neural network which was build based on <a href="https://www.youtube.com/watch?v=1lwddP0KUEg" target="_blank">this tutorial</a>. If prediction accuracy is below 85%, a German version of <a href="https://github.com/wadetb/eliza" target="_blank">ELIZA</a> (a program that pretends to be a Rogerian psychologist) is activated instead. </li>
<li>The bot is able to recognize location names using <b>spaCy</b>. If location does not belong to the list of German states, user is referred to the WHO website for further information.</li>
</ul>

More details as to how the chatbot was designed, can be found in the following <a href="https://github.com/valeriiabubela/dialogsystem/blob/69ce8e0fe76d1f9024c261e2fb5abe5730e13743/Dialogsystem_Bubela.ipynb" target="_blank">jupyter notebook</a>.
