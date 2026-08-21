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

    initProjectRail();
})

// Horizontal project rail: drag with the mouse, scroll with wheel/trackpad,
// swipe on touch, or step with the chevron buttons.
function initProjectRail() {
    var rail = document.querySelector('.project-rail');
    var track = rail && rail.querySelector('.project-rail-track');
    if (!track) {
        return;
    }

    var prevBtn = rail.querySelector('.rail-nav.is-prev');
    var nextBtn = rail.querySelector('.rail-nav.is-next');

    // --- edge fades + button visibility -------------------------------------
    var EDGE_TOLERANCE = 8;  // sub-pixel layout and snap offsets, in px

    function updateAffordances() {
        var maxScroll = track.scrollWidth - track.clientWidth;
        var atStart = track.scrollLeft <= EDGE_TOLERANCE;
        var atEnd = track.scrollLeft >= maxScroll - EDGE_TOLERANCE;
        var scrollable = maxScroll > EDGE_TOLERANCE;

        rail.classList.toggle('has-fade-start', scrollable && !atStart);
        rail.classList.toggle('has-fade-end', scrollable && !atEnd);

        if (prevBtn) {
            prevBtn.hidden = !scrollable || atStart;
        }
        if (nextBtn) {
            nextBtn.hidden = !scrollable || atEnd;
        }
    }

    track.addEventListener('scroll', updateAffordances, { passive: true });
    window.addEventListener('resize', updateAffordances);
    updateAffordances();
    // a card expanding via "Read more" changes the rail's scrollable width
    $('.read-more-btn').click(function() {
        window.setTimeout(updateAffordances, 0);
    });

    // --- step buttons --------------------------------------------------------
    function stepSize() {
        var item = track.querySelector('.project-rail-item');
        if (!item) {
            return track.clientWidth * 0.8;
        }
        var gap = parseFloat(window.getComputedStyle(track).columnGap) || 24;
        return item.getBoundingClientRect().width + gap;
    }

    if (prevBtn) {
        prevBtn.addEventListener('click', function() {
            track.scrollBy({ left: -stepSize(), behavior: 'smooth' });
        });
    }
    if (nextBtn) {
        nextBtn.addEventListener('click', function() {
            track.scrollBy({ left: stepSize(), behavior: 'smooth' });
        });
    }

    // --- drag to pan ---------------------------------------------------------
    var pointerId = null;
    var startX = 0;
    var startScroll = 0;
    var isDragging = false;   // past the threshold, actually panning
    var justDragged = false;  // swallow the click that ends a drag

    var DRAG_THRESHOLD = 5;   // px before a press counts as a drag, not a click

    track.addEventListener('pointerdown', function(e) {
        // Touch and pen keep their native scrolling/swipe; this is the mouse path.
        if (e.pointerType !== 'mouse' || e.button !== 0) {
            return;
        }
        pointerId = e.pointerId;
        startX = e.clientX;
        startScroll = track.scrollLeft;
        isDragging = false;
    });

    track.addEventListener('pointermove', function(e) {
        if (pointerId === null || e.pointerId !== pointerId) {
            return;
        }
        var dx = e.clientX - startX;
        if (!isDragging) {
            if (Math.abs(dx) < DRAG_THRESHOLD) {
                return;
            }
            isDragging = true;
            track.classList.add('is-dragging');
            try {
                track.setPointerCapture(pointerId);
            } catch (err) { /* capture is a nicety, not a requirement */ }
        }
        track.scrollLeft = startScroll - dx;
        e.preventDefault();
    });

    function endDrag(e) {
        if (pointerId === null || (e && e.pointerId !== pointerId)) {
            return;
        }
        if (isDragging) {
            track.classList.remove('is-dragging');
            try {
                track.releasePointerCapture(pointerId);
            } catch (err) { /* nothing to release */ }
            // the click generated by this pointerup must not open a card action
            justDragged = true;
            window.setTimeout(function() {
                justDragged = false;
            }, 0);
        }
        pointerId = null;
        isDragging = false;
    }

    track.addEventListener('pointerup', endDrag);
    track.addEventListener('pointercancel', endDrag);
    track.addEventListener('lostpointercapture', endDrag);

    track.addEventListener('click', function(e) {
        if (justDragged) {
            e.preventDefault();
            e.stopPropagation();
        }
    }, true);

    // keep the browser's own image/text drag out of the way
    track.addEventListener('dragstart', function(e) {
        e.preventDefault();
    });
}
