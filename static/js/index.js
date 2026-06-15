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

    // Toggle main downstream accordion
    $('.downstream-main-toggle').click(function() {
        var accordion = $(this).closest('.downstream-accordion');
        var content = accordion.find('.downstream-main-content');
        
        accordion.toggleClass('is-active');
        content.slideToggle(300);
        
        // If main is closed, close all nested items to keep it clean for next time
        if (!accordion.hasClass('is-active')) {
            $('.nested-accordion-item').removeClass('is-active');
            $('.nested-accordion-content').slideUp(0);
            $('.bibtex-content-box').slideUp(0);
        }
    });

    // Toggle nested accordion items (mutually exclusive)
    $('.nested-accordion-header').click(function() {
        var item = $(this).closest('.nested-accordion-item');
        var content = item.find('.nested-accordion-content');
        var siblingItems = item.siblings('.nested-accordion-item');
        
        // Close siblings
        siblingItems.removeClass('is-active');
        siblingItems.find('.nested-accordion-content').slideUp(300);
        siblingItems.find('.bibtex-content-box').slideUp(300);
        
        // Toggle current
        item.toggleClass('is-active');
        content.slideToggle(300);
    });

    // Toggle BibTeX inline views
    $('.bibtex-toggle-btn').click(function() {
        var bibtexBox = $(this).closest('.nested-project-details').find('.bibtex-content-box');
        bibtexBox.slideToggle(200);
    });

})
