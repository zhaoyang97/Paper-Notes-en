// Make the header title clickable → navigate to homepage
document.addEventListener("DOMContentLoaded", function () {
  var title = document.querySelector(".md-header__title");
  if (title) {
    title.style.cursor = "pointer";
    title.addEventListener("click", function () {
      window.location.href = document.querySelector(".md-header__button.md-logo")
        ? document.querySelector(".md-header__button.md-logo").href
        : "./";
    });
  }
});
