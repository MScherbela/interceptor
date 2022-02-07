<template>
  <b-card class="py-0 titlecard cardbackground" body-class="py-0">
    <b-card-title class="my-1"><h1 class="my-0"><b>{{ title }}</b></h1></b-card-title>
  </b-card>
</template>

<script>
export default {
  name: "TitleCard",
  props: {
    title: String
  }
}
</script>

<style scoped>

@import url('https://fonts.googleapis.com/css2?family=UnifrakturCook:wght@700&display=swap');

.titlecard {
  font-family: UnifrakturCook;
  filter: brightness(0.8);
}

</style>
