jQuery(document).ready(function($) {

    function myRetina(){
        var isRetina = (
            window.devicePixelRatio > 1 ||
            (window.matchMedia && window.matchMedia("(-webkit-min-device-pixel-ratio: 1.5),(-moz-min-device-pixel-ratio: 1.5),(min-device-pixel-ratio: 1.5)").matches)
            );
        if(isRetina && ($('#thelogo a img').attr('data-status') != 'ok') && ($('#thelogo a img').attr('data-retina') != $('#thelogo a img').attr('data-first'))){            
            var dataretina=$('#thelogo a img').attr('data-retina'); 
            if(dataretina == $('#thelogo a img').attr('src')){
                if (($(window).width() < 767) || ($(window).width() > 991)) {
                    $('#thelogo a img').attr('width',$('#thelogo a img').width()/2);
                    $('#thelogo a img').attr('height',$('#thelogo a img').height()/2);
                    $('#thelogo a img').attr('data-status','ok');
                }
            }else{           
                $('#thelogo a img').attr('src',dataretina).load(function(){            
                    if (($(window).width() < 767) || ($(window).width() > 991)) {
                        this.width=this.width/2;
                        this.height=this.height/2;
                        $('#thelogo a img').attr('data-status','ok');
                    }
                });
                var dataretinasmall=$('#thelogosmall a img').attr('data-retina');
                $('#thelogosmall a img').attr('src',dataretinasmall).load(function(){
                    this.width=this.width/2;
                    this.height=this.height/2;
                });
            }
        }
    }
    myRetina();
    $( window ).resize(function() {        
       myRetina();
    });
    // Ajax Search
    $( ".search-field" ).keyup(function() {   
        if ($('div').hasClass('search-result')){   
          var search_val=$(".search-field").val(); 
          if(search_val!=""){
            $('.loading-search-result').show();
            $.ajax({
              type:"POST",
              url: MahaAjax.ajaxurl,
              data: {
                action:'maha_ajax_search', 
                search_string:search_val,
                load:'ok'
            },
            success:function(data){
                $('.search-result-content').html(data);
                $('.search-result').show();
                $('.loading-search-result').hide();
            }
            });  
            }else{
                $('.search-result').hide();
            } 
        }
    });

    // Infinite Scroll
    $('#cur-page').infinitescroll({
        navSelector  : ".a-nav",            
        nextSelector : ".a-next a",  
        itemSelector : ".cur-page-item",
        debug        : false
    },function(){
        var $el_block_2 = $('.el-block-1, .el-block-2, .el-block-4, .el-module-2, .el-block-6, .el-block-7');
        $el_block_2.imagesLoaded().done(function() {
            setTimeout(function(){
                $el_block_2.isotope();
            },1000);        
        }).fail(function() {
            setTimeout(function(){
                $el_block_2.isotope();
            },3000);        
        });
    });

    // Running Text   
    var jmlthumb=0;
    var first_runtext=$('.myrun').html();
    $('.thumb-runtext').each(function (index,element){       
        jmlthumb+=1;                 
    });
    
    function resize_running_text(){                       
        if($('#runtext').length == 0){
            $('.myrun > div:first-child').remove();
            $('.myrun').append(first_runtext);
        }        
        
        if($(window).width() <= 767){
            $('.myrun > div:first-child').remove();
        }else{
            var crtwidth=$('.container').width();
            var current_item_container=0;     
            $('.thumb-runtext').each(function (index,element){ 
                current_item_container+=$(this).outerWidth(true);               
            }); 
            if(crtwidth>current_item_container){
                var sisa=crtwidth-current_item_container;
                var tambahan=sisa/jmlthumb;
                $('.thumb-runtext').each(function (index,element){ 
                    var newthumb=$(this).width()+tambahan;
                    $(this).width(newthumb);
                });
            }  

            marqueeInit({
             uniqueid: 'runtext',
             style: {
              'margin-top': '5px',
              'margin-bottom': '20px',
              'overflow': 'hidden',
              'white-space':'nowrap',
              'width':'100%'
          },
            inc: 5, //speed - pixel increment for each iteration of this marquee's movement
            mouse: 'cursor driven', //mouseover behavior ('pause' 'cursor driven' or false)
            moveatleast: 1,
            neutral: 150,
            direction: 'left',
             savedirection: true
        });
        }
    }    

    resize_running_text();           
    $( window ).resize(function() {        
        resize_running_text();     
    });


    // Mobile Nav
    $('.mobile-bar ul li').each(function(index, element) {
        if ($(element).children().length === 3) {
            $(this).children('span').addClass('navmob-sub-menu').html('<i class="icon-down-open-mini"></i>');
        }
    });
    $('.mobile-bar ul li .navmob-sub-menu').click(function() {
        $(this).next().next().slideToggle(300);
        if ($(this).children().hasClass('icon-down-open-mini')) {
            $(this).children().removeClass('icon-down-open-mini').addClass('icon-up-open-mini');
        } else {
            $(this).children().removeClass('icon-up-open-mini').addClass('icon-down-open-mini');
        }
    });

    // Off Canvas
    var SidebarMenuEffects = (function() {

        function hasParentClass(e, classname) {
            if (e === document)
                return false;
            if (classie.has(e, classname)) {
                return true;
            }
            return e.parentNode && hasParentClass(e.parentNode, classname);
        }

        // mobile check - coveroverflow-com/a/11381730/989439
        function mobilecheck() {
            var check = false;
            (function(a) {
                if (/(android|ipad|playbook|silk|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows (ce|phone)|xda|xiino/i.test(a) || /1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0, 4)))
                    check = true
            })(navigator.userAgent || navigator.vendor || window.opera);
            return check;
        }

        function init() {

            var container = document.getElementById('body-maha'),
                body = document.body,
                buttons = Array.prototype.slice.call(document.querySelectorAll('#top-mobile-wrapper ul li a')),
                // event type (if mobile use touch events)
                eventtype = mobilecheck() ? 'touchstart' : 'click',
                resetMenu = function() {
                    classie.remove(container, 'st-menu-open');
                    classie.remove(body, 'canvas-on');
                },
                bodyClickFn = function(evt) {
                    if (!hasParentClass(evt.target, 'mobile-bar')) {
                        resetMenu();
                        document.removeEventListener(eventtype, bodyClickFn);
                    }
                };

            buttons.forEach(function(el, i) {
                var effect = 'st-effect-4';

                el.addEventListener(eventtype, function(ev) {
                    ev.stopPropagation();
                    ev.preventDefault();
                    container.className = 'body-maha'; // clear
                    classie.add(container, effect);
                    setTimeout(function() {
                        classie.add(container, 'st-menu-open');
                        classie.add(body, 'canvas-on');
                    }, 25);
                    document.addEventListener(eventtype, bodyClickFn);
                });
            });

            document.getElementById('close-mobile-bar').onclick = function() {
                resetMenu();
                document.removeEventListener(eventtype, bodyClickFn);
            };
        }

        init();

    })();

    $('body.ie #top-mobile-wrapper ul li a').click(function() {
        $('body.ie').addClass('canvas-on');
        $('body.ie .body-maha').addClass('st-menu-open');
    });

    // Search Header
    $('.open-search-form').click(function() {
        //$('.main-nav-bar').addClass('search-form');
        $('#con-search').fadeToggle('200', function() {
            $('#main-search-form .search-field').focus();
        });
    });
    $('.close-search-form').click(function() {
        $('#con-search').fadeOut(200);
        //$('.main-nav-bar').removeClass('search-form');
    });

    $('#cart-nav').mouseenter(function(){
        $('#shopping-cart-widget').stop().fadeIn(400);
    });
    $('#cart-nav').mouseleave(function(){
        $('#shopping-cart-widget').stop().fadeOut(200);
    });

    // Main Menu
    $('.main-ul-nav li .nav-sub-menus').not($('.nav-sub-wrap .nav-sub-menus')).wrap('<div class="nsw" />');
    $('.main-ul-nav li .nav-sub-wrap .nsw').each(function(index, element) {
        if ($(element).children().length === 1) {
            if ($(element).children().attr('class') == 'nav-sub-menus') {
                $(element).parents('li').addClass('ord-nav-offset');
                $(element).parent().addClass('ord-nav').removeClass('container');
                $(element).removeClass('row');
            } else {
                $(element).addClass('mm-full');
            }
        }

        if ($(element).children().length === 0) {
            $(element).parent().remove();
        }

        if ($(element).children().length === 2) {
            $(element).find('.nav-sub-posts').addClass('col-sm-9');
            $(element).find('.nav-sub-posts .col-sm-3:last-child').remove();
            $(element).find('.nav-sub-posts .col-sm-3').addClass('col-sm-4').removeClass('col-sm-3');
            $(element).find('.nav-sub-menus').addClass('col-sm-3').insertBefore($(element).find('.nav-sub-posts'));
            $(element).find('.nav-sub-menus:not(:first-child)').remove();
        }
    });

    $('.nav-sub-menus > ul > li > .nav-sub-menus').each(function(index, element) {
        $(element).wrap('<div class="nav-sub-wrap ord-nav" />');
        $(element).wrap('<div class="nsw" />');
    });

    $('.main-ul-nav > ul > li > .submenu-languages').each(function(index, element) {
        $(element).wrap('<div class="nav-sub-wrap ord-nav" />');
        $(element).wrap('<div class="nsw" />');
        $(element).wrap('<div class="nav-sub-menus" />');
    });

    $('.main-ul-nav li').each(function(index, element) {
        var sub_ord_i = 'icon-right-open-mini';
        if ($(element).children('.nav-sub-wrap').length > 0) {
            if ($(element).parent().parent().hasClass('main-ul-nav')) {
                sub_ord_i = 'icon-down-open-mini';
            } else {
                sub_ord_i = 'icon-right-open-mini';
            }
            $(element).children('a').append('<span class="sub-ord-nav"> <i class="' + sub_ord_i + '"></i> </span>');
        }
    });

    $('.main-ul-nav li').mouseenter(function() {
        $(this).find('> .nav-sub-wrap').show().animate({opacity: 1}, 130);
    }).mouseleave(function() {
        $(this).find('> .nav-sub-wrap').animate({opacity: 0}, 130).hide();
    });

    // Moz Slider
    $('.wrap-moz-item').mouseenter(function() {
        $(this).find('.zoom-it .detail').addClass('zoom-detail').prependTo($(this)).animate({'padding-bottom': 20}, 210);
    }).mouseleave(function() {
        $(this).find('.zoom-detail').animate({'padding-bottom': 17}, 230, function() {
            $(this).appendTo($(this).next('.zoom-it')).removeClass('zoom-detail');
        });
    });

    // Moz Clicked
    var isDragging = false;
    $(".moz-item").mousedown(function(e) {
        // isDragging = false;
        // console.log(e);
        $(window).mousemove(function(e) {
            isDragging = true;
            // console.log(isDragging);
            $(window).unbind("mousemove");
        });
    }).mouseup(function(e) {
        // console.log(e.which);
        if (e.which === 1) {
            var wasDragging = isDragging;
            isDragging = false;
            $(window).unbind("mousemove");
            if (!wasDragging) {
                window.location.href = $(this).find('.detail a.moz-url').attr('href');
            }
        }
        if (e.which === 2) {
            var wasDragging = isDragging;
            isDragging = false;
            $(window).unbind("mousemove");
            if (!wasDragging) {
                window.open($(this).find('.detail a.moz-url').attr('href'));
                // window.location.href = $(this).find('.detail a.moz-url').attr('href');
            }
        }
    });

    // Scroll to TOP Action
    var scroll_move = false,
            off_canvas = $('#off-canvas-body'),
            main_nav_bar = $('#main-nav-bar'),
            main_nav_offset = $(main_nav_bar).offset().top,
            main_nav_height = $(main_nav_bar).outerHeight();
    admin_bar = 0;

    if ($('#wpadminbar').length > 0) {
        admin_bar = 28;
    }

    $(document).bind('mousewheel DOMMouseScroll MozMousePixelScroll', function(e) {
        if (scroll_move === false) {
            return;
        } else {
            scroll_move = false;
            $("html, body").stop();
        }
    });

    if (document.addEventListener) {
        document.addEventListener('touchmove', function(e) {
            if (scroll_move === false) {
                return;
            } else {
                scroll_move = false;
                $("html, body").stop();
            }
        }, false);
    }

    $(window).scroll(function() {

        if ($(main_nav_bar).css('display') != 'none') {
            if (main_nav_offset == 0) {
                main_nav_offset = $('.main-logo-ads-wrap').offset().top + $('.main-logo-ads-wrap').outerHeight();
            }
            if ($(this).scrollTop() > main_nav_offset) {
                $(main_nav_bar).addClass('on-stuck');
                $('.on-stuck').css('margin-top', admin_bar);
                $('#con-search').css('top', 56);
                $(off_canvas).css('margin-top', main_nav_height);
                $('#thelogosmall').show();

            } else {
                $(main_nav_bar).css('margin-top', 0).removeClass('on-stuck');
                $(off_canvas).css('margin-top', 0);
                $('#con-search').css('top', 57);
                $('#thelogosmall').hide();
            }
        }

        if (scroll_move) {
            return;
        }

        if ($(this).scrollTop() > 400) {
            $('#scrolltop').addClass('scroll-up');
        } else {
            $('#scrolltop').removeClass('scroll-up');
        }

    });

    $('#scrolltop').click(function(e) {
        $("html, body").animate({scrollTop: 0}, {
            duration: 1200,
        });
        e.preventDefault();
    });

    // Loop Blocked Posts Isotope
    var $el_block_2 = $('.el-block-1, .el-block-2, .el-block-4, .el-module-2, .el-block-6, .el-block-7');
    $el_block_2.imagesLoaded().done(function() {
        setTimeout(function(){
            $el_block_2.isotope();
        },1000);        
    }).fail(function() {
        setTimeout(function(){
            $el_block_2.isotope();
        },3000);        
    });

    // Single Post Layout Parallax
    $('.i-parallax').each(function() {
        var $bgobj = $(this); // assigning the object

        $(window).scroll(function() {

            // Put together our final background position
            if ($bgobj.hasClass('with-fp')) {
                var yPos = ($(window).scrollTop() * 0.08);
                var coords = '50% ' + (93 - yPos) + '%';
            } else {
                var yPos = ($(window).scrollTop() * 0.13);
                var coords = '50% ' + (80 - yPos) + '%';
            }

            // Move the background
            $bgobj.css({backgroundPosition: coords});
        });
    });

    // Review Circle
    $(".dial").knob();

    // Format Gallery
    if ($('.cover .cf-gallery').length) {

        var cfgallery_num = $('.cover .cf-gallery .cf-item').size();
        var currimg = 0;

        function loadimg() {

            $('.cover').animate({opacity: 1}, 100, function() {

                //finished animating, minifade out and fade new back in           
                $('.cover').animate({opacity: 0.2}, 300, function() {

                    currimg++;
                    $(this).find('.cf-item').removeClass('active');
                    $(this).find('.cf-item:eq(' + (currimg - 1) + ')').addClass('active');

                    if (currimg > cfgallery_num - 1) {
                        currimg = 0;
                    }

                    var newimage = $('.cf-item.active img').attr('src');

                    $(this).css("background-image", "url(" + newimage + ")");

                    $(this).animate({opacity: 1}, 400, function() {
                        setTimeout(loadimg, 5000);
                    });

                });

            });

        }
        setTimeout(loadimg, 5000);
    }

    // For Post Type Regular
    $('.mini-gallery').maha_royalSlider({
        arrowsNav: true,
        arrowsNavAutoHide: false,
        fadeinLoadedSlide: false,
        controlNavigationSpacing: 0,
        imageScaleMode: 'fill',
        imageAlignCenter: false,
        blockLoop: true,
        loop: true,
        numImagesToPreload: 5,
        autoScaleSlider: false,
        autoHeight: true,
        transitionType: 'fade',
        keyboardNavEnabled: true,
        transitionSpeed: 900,
        autoPlay: {
            enabled: true,
            pauseOnHover: true,
            delay: 4700
        },
        block: {
            delay: 2500
        }
    });

    // Review Bar
    $('.maha-progress-bar').each(function(i) {

        $(this).appear(function() {

            var percent = $(this).find('.bar').attr('data-width');
            var $that = $(this);

            $(this).find('.bar').animate({
                'width': percent + '%'
            }, 1600, 'easeOutCirc');

            //100% progress bar 
            if (percent == '100') {
                $that.find('.bar').addClass('full');
            }

        });

    });

    // Play the Video Button - Hover
    $('.play-the-media').hover(function() {
        $('.cover').animate({opacity: 0.7}, 300);
    }, function() {
        $('.cover').animate({opacity: 1}, 300);
    });
    $('.play-the-media').click(function() {
        var media_url = $(this).attr('data-media');
        $('.cover').animate({'bottom': '-1000px'}, 500).hide(300).animate({'bottom': '0px'});

        $(this).animate({opacity: 0}, 300).hide(300);

        if ($(this).hasClass('video')) {
            $('.player-wrap').show(900).find('.row .col-sm-12').append(media_url.replace(/(?:http:\/\/|https:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?(.+)/g, '<iframe width="1142" height="658" src="//www.youtube.com/embed/$1&autoplay=1" frameborder="0" allowfullscreen></iframe>').replace(/(?:http:\/\/|https:\/\/)?(?:www\.)?(?:vimeo\.com)\/(.+)/g, '<iframe src="//player.vimeo.com/video/$1" width="1142" height="658" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>').replace(/(?:http:\/\/|https:\/\/)?(?:www\.)?(?:dailymotion\.com)\/(.+)/g, '<iframe width="1142" height="658" src="//www.dailymotion.com/embed/$1&autoPlay=1" frameborder="0" allowfullscreen></iframe>')).animate({opacity: 1}, 1100);
            $('.player-wrap').fitVids();
        } else if ($(this).hasClass('audio')) {
            $('.player-wrap').show(900);
            $.getJSON('//soundcloud.com/oembed?callback=?',
                    {format: 'js', url: media_url, iframe: true, auto_play:true},
            function(data) {
                $('.player-wrap').find('.row .col-sm-12').append(data['html']);
            }
            )
        }        
        $('.player-wrap').animate({opacity: 1}, 700);
    });    

    $('.back-nav').click(function(){
        $('.cover').show(300);
        $('.player-wrap').hide(300).find('.row .col-sm-12 .fluid-width-video-wrapper').remove();
        $('.player-wrap').hide(300).find('.row .col-sm-12 iframe').remove();
        $('.play-the-media').animate({opacity:1},300).show(300);
    });

    // Responsive Video 
    $('.video-wrapper').fitVids();
    $('.video-wrapper').fitVids({ customSelector: "iframe[src^='http://dailymotion.com'], iframe[src^='http://www.dailymotion.com'], iframe[src^='https://www.dailymotion.com']"});

    // Animate
    if ($('.off-canvas-body').hasClass('animati-on')) {

        // console.log($(window).width());
        if ($(window).width() < 768) {
            $('.up-up, .up-up-child > div').removeClass('animated fadeInUp');
            $('.up-up, .up-up-child > div').addClass('no_animated');
        } else {
            $('.up-up, .up-up-child > div').removeClass('no_animated');
            $('.up-up, .up-up-child > div').data('appear-top-offset', 700).appear(function() {
                $(this).addClass('animated fadeInUp');
            });
        }

        $(window).resize(function() {
            if ($(window).width() < 768) {
                $('.up-up, .up-up-child > div').removeClass('animated fadeInUp');
                $('.up-up, .up-up-child > div').addClass('no_animated');
            } else {
                $('.up-up, .up-up-child > div').removeClass('no_animated');
                $('.up-up, .up-up-child > div').addClass('animated fadeInUp');
            }
        });
    }

    // Widget Popular
    $(window).resize(function() {
        $('.widget_popular_post').each(function(){
            var height = $(this).find(' .popular-show').height();
            $(this).find(' .popular').css('height', height); 
        });                    
    });

    $('.widget_popular_post .nav-popular-post ul li a').on('click', function() {
        var current = $(this);
        var curclass = current.attr('class').replace('popular-show','').replace('popular-active','').replace('-','_');
        var height = current.closest('.widget_popular_post').find('.popular .'+curclass).height();
        current.closest('ul').find('a').removeClass('popular-active');
        current.addClass('popular-active');
        current.closest('.widget_popular_post').find('div.popular > div').removeClass('popular-show');        
        current.closest('.widget_popular_post').find('.popular .'+curclass).addClass('popular-show');               
        current.closest('.widget_popular_post').find('.popular').css('height', height);
        current.closest('.widget_popular_post').find('.popular > div').css({'opacity': 0, 'display': 'none'});
        setTimeout(function() {
            current.closest('.widget_popular_post').find('.popular .'+curclass).css({'opacity': 1, 'display': 'block'});
        }, 300);
        return false;
    }); 

    // WP Caption
    var img_caption = $(".wp-caption");
    $('.wp-caption').each(function(i) {
        var img_src = $(this).find('a').attr('href');
        $(this).removeAttr('style');
        if (img_src.indexOf('attachment_id') != -1) {
            // nothing
        } else {
            $(this).children('a').removeAttr('href');
            $(this).append('<figure></figure>');
            $(this).children('a').appendTo($(this).find('figure').attr('data-src', img_src));
            $(this).children('p.wp-caption-text').appendTo($(this).find('figure'));
            $(this).lightGallery();
        }
    });

    // Login Modal
    $('.login-modal-closer').click(function(){$('#cur-login').hide();});
    $('.register-modal-closer').click(function(){$('#cur-register').fadeOut();});
    $('.remember-modal-closer').click(function(){$('#cur-remember').fadeOut();});


    // Woocommerce
    if ( $('.woocommerce ul.products').length > 0 ) {
        var $woo_products = $('.woocommerce ul.products');
        $woo_products.imagesLoaded(function() {
            $woo_products.isotope({
                layoutMode: 'fitRows'
            });
            $woo_products.imagesLoaded().done(function() {
                setTimeout(function(){
                    $woo_products.isotope({
                       layoutMode: 'fitRows'
                    });
                },1000);                                          
            }).fail( function() {
                setTimeout(function(){
                    $woo_products.isotope({
                       layoutMode: 'fitRows'
                    });
                },3000);
            });
        });

        // var $woo_item = $('.woocommerce ul.products > li');
        // $woo_item.each( function(i){
        //     var $this_item = $(this);
        //     $(this).find('.maha-product-wrap').prepend($(this).find('.onsale').delay(i*130).animate({'opacity':1},350).parent().addClass('onsale_wrap'));
        // });
    }

    /* -------------------------------------------------------------
    Google Map
*/
    function initialize() {
        var LatLng = $('#map-canvas').data('point').split(',');
        var myLatlng = new google.maps.LatLng(LatLng[0], LatLng[1]);
        var mapOptions = {
            zoom: $('#map-canvas').data('zoom'),
            center: myLatlng
        }
        var map = new google.maps.Map(document.getElementById('map-canvas'), mapOptions);

        var marker = new google.maps.Marker({
            position: myLatlng,
            map: map,
        });
    }

    if($('#map-canvas').length){
        google.maps.event.addDomListener(window, 'load', initialize);
    }
        
});