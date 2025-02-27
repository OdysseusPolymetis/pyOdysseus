/**
 * Pilote de l‘application javascript Odysseus
 * Pas de dépendance à librairie javascript
 */

// différentes déclarations globables pour simplifie les écritures
if (!window.XMLHttpRequest && 'ActiveXObject' in window) {
  window.XMLHttpRequest= function() {
    return new ActiveXObject('MSXML2.XMLHttp');
  };
}
Element.prototype.remove = function() {
  this.parentElement.removeChild(this);
}
NodeList.prototype.remove = HTMLCollection.prototype.remove = function() {
  for(var i = this.length - 1; i >= 0; i--) {
    if(this[i] && this[i].parentElement) {
      this[i].parentElement.removeChild(this[i]);
    }
  }
}

// Création conditionnelle de l’objet global
var Odysseus = Odysseus || {};
// Identifiant du div contenant les textes, sera remplacé par un pointeur DOM
Odysseus.slider = "slider";
Odysseus.last = [];



// Appelé au chargement de la page, event document.onload
Odysseus.load = function() {
  // si un élément logger dans le document HTML, servira à sortir des messages
  Odysseus.logger = document.getElementById( "logger" );
  // enregistrer le mouseover sur les chunks
  Odysseus.slider = document.getElementById( Odysseus.slider );
  if ( ! Odysseus.slider ) return;
  // observer le survol sur les colonnes pour aligner
  Odysseus.slider.onmouseover = Odysseus.segover;
  // peupler les options du sélecteur de texte
  var select = document.getElementById( "seltext" );
  var dates = []; // pour un tri en ordre de dates
  for ( id in Odysseus.corpus ) {
    if ( !Odysseus.corpus.hasOwnProperty(id) ) continue; // passer des trucs du genre toString()
    var option = document.createElement("option");
    option.value = id;
    option.text = Odysseus.corpus[id].creator+", "+Odysseus.corpus[id].date;
    select.add(option);
    // ici on peuple le tableau à trier sur la date
    if(!Odysseus.corpus[id].date) Odysseus.corpus[id].date = 0000;
    dates.push(""+Odysseus.corpus[id].date+id);
  }
  dates.sort();
  // maintenant on ajoute les textes en ordre de dates
  for ( i in dates ) {
    console.log(dates[i]);
    var id = dates[i].substring(4);
    Odysseus.textadd( id );
  }

};
// Enregistrer le chunk survolé, peut servir à d'autres événements (ex: scroll)
Odysseus.segover = function( e ) {
  if (!e) var e = window.event;
  var seg = e.relatedTarget || e.fromElement;
  while ( seg && !Odysseus.classHas( seg, "chunk" ) ) {
    seg = seg.parentNode;
    if ( !seg ) return;
  }
  if ( seg == Odysseus.segread ) return;
  Odysseus.segread = seg;
  Odysseus.aligncols( seg );
};

// Aligner les colonne à partir de la position d‘un chunk
Odysseus.aligncols = function( source ) {
  if ( !source ) return;
  if ( !source.id ) {
    console.log( source );
    return;
  }
  // prendre le numéro d'ordre du chunk
  var suf = source.id.substring( source.id.lastIndexOf( '-' )+1 );
  // attraper le scrolleur parent (offsetParent ne donne pas ce qu'il faut)
  col = Odysseus.scroller( source );
  var diffy = 0;
  diffy = source.offsetTop - col.scrollTop;
  // boucler sur les colonnes supprimer la sélection
  if ( Odysseus.last && Odysseus.last.length ) {
    for ( var i = 0; i < Odysseus.last.length; i++) {
      Odysseus.classDel( Odysseus.last[i], "selected" );
    }
  }
  Odysseus.last = [];
  var cols = Odysseus.slider.getElementsByTagName("div");
  for ( var i=0; i < cols.length; i++ ) {
    // attraper l’identifiant de text
    var section = cols[i].getElementsByTagName("section");
    if ( !section || !section.length ) continue;
    var chapterid = section[0].id;
    var segid = chapterid+'-'+suf;
    var target = document.getElementById( segid );
    Odysseus.classAdd( target, "selected" );
    Odysseus.last.push( target );
    if ( segid == source.id ) continue; // on ne bouge pas la source;
    if ( !target ) continue;
    col = Odysseus.scroller( target );
    // target.scrollIntoView(true);
    col.scrollTop = target.offsetTop - diffy;
  }
};

// Depuis n’importe quel poin du document DOM, attraper le premier conteneur avec du scroll
Odysseus.scroller = function( el ) {
  while (true) {
    if ( !el ) return false;
    if (el.className == "col") return el;
    el = el.parentNode;
  }
};
// l’élément contient-il cette classe CSS ?
Odysseus.classHas = function (el, cls) {
  if ( !el ) return false;
  if ( el.className ) var cls2 = '  '+el.className+' ';
  else var cls2 = '  '+el+' ';
  return ( cls2 ).indexOf( ' '+cls+' ' ) > -1;
};
// retirer une classe CSS ?
Odysseus.classDel = function (el, cls) {
  if ( !el ) return el;
  if ( typeof el === "string" ) var cls2 = '  ' + el + ' ';
  else if ( typeof el === "object" ) var cls2 = '  '+el.className+' ';
  else return el;
  var re = new RegExp(" "+cls+" ", "g");
  cls2 = cls2.replace( re, ' ' ).trim();
  if ( typeof el === "string" ) return cls2;
  el.className = cls2;
  return el;
};
// Ajouter une classe CSS
Odysseus.classAdd = function (el, cls) {
  el = Odysseus.classDel( el, cls );
  el.className += " "+cls;
  return el;
};
// un log maison pour le débogage
Odysseus.log = function ( message ) {
  if ( !Odysseus.logger ) return;
  Odysseus.logger.innerHTML = message;
};
// Afficher le contenu html d'un fichier dans une div
Odysseus.showfile = function( div, textid, chapterid, stats ) {
  var href = "html/"+textid+'_'+chapterid+'.html';
  var header = '<header><button type="button" onclick="this.parentNode.parentNode.remove()">❌</button><h1>'+Odysseus.corpus[textid].date+', '+Odysseus.corpus[textid].creator+', chant '+chapterid+'</h1></header>';
  var footer = '<footer></footer>';
  var xhr= new XMLHttpRequest();
  xhr.open( 'GET', href, true );
  xhr.onreadystatechange = function() {
    if ( this.readyState !== 4 ) return;
    if ( this.status!==200 ) {
      div.innerHTML= header+"<p>Chant absent.</p>"+footer;
      return;
    }
    div.innerHTML= header+this.responseText+footer;
  };
  xhr.send();
};
// Ajouter un texte à la suite
Odysseus.textadd = function( textid ) {
  if ( !textid ) return;
  console.log( textid );
  var sel = document.getElementById('selchapter');
  var chapterid = sel.options[ sel.selectedIndex ].value;
  var stats = textid;
  var stats = document.getElementById('chapterid');
  if ( !chapterid ) chapterid = 1;
  var col = document.createElement("div");
  col.onscroll = Odysseus.colscroll;
  col.className = "col";
  col.textid = textid;
  
  Odysseus.slider.appendChild( col );
  Odysseus.showfile( col, textid, chapterid, stats );
};
//  Scrollage d‘une colonne’
Odysseus.colscroll = function() {
  if ( !this ) return;
  var header = this.getElementsByTagName( "header" )[0];
  header.style.top = this.scrollTop+"px";
};
// Selectionner une section
Odysseus.selchapter = function( chapterid ) {
  var cols = Odysseus.slider.getElementsByTagName("div");
  for ( var i=0; i < cols.length; i++ ) {
    col = cols[i];
    if ( !col.textid ) continue;
    Odysseus.showfile( col, col.textid, chapterid );
  }
};
Odysseus.stats = function( chapterid ) {
	  var cols = Odysseus.slider.getElementsByTagName("section");
	  for ( var i=0; i < cols.length; i++ ) {
	    col = cols[i];
	    if ( !col.textid ) continue;
	    Odysseus.showfile( col, col.textid, chapterid );
	  }
	};
// Par défaut, fichier charger en fin de HTML, on charge tous les textes
Odysseus.load();