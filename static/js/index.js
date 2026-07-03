window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

    // Expand / collapse the full description of a project card
    $('.read-more-btn').click(function() {
        var btn = $(this);
        var card = btn.closest('.project-card');
        var expanded = card.toggleClass('is-expanded').hasClass('is-expanded');

        btn.toggleClass('is-expanded');
        btn.find('.read-more-label').text(expanded ? 'Read less' : 'Read more');
    });

    // Toggle BibTeX inline views
    $('.bibtex-toggle-btn').click(function() {
        var bibtexBox = $(this).closest('.project-card').find('.bibtex-content-box');
        bibtexBox.slideToggle(200);
    });

})
