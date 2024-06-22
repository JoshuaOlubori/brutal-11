import React, { useState, useEffect } from 'react';
import { getCollection } from "astro:content";
// import { Card } from '@eliancodes/brutal-UI'; // Assuming Brutal UI components are accessible
// import { Button } from '@eliancodes/brutal-UI';
// import BlogList from '@components/blog/BlogList'; // Assuming BlogList is a separate React component
import BrutalPill from '@components/generic/BrutalPill';

import BrutalCard from "@components/generic/BrutalCard";
import BrutalButton from '@components/generic/BrutalButton';


const RecentBlogPosts = () => {
  const [posts, setPosts] = useState([]);
  const [categories, setCategories] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('All');

  useEffect(() => {
    const fetchData = async () => {
      const fetchedPosts = await getCollection('blog'); // Assuming getCollection is accessible
      const formattedPosts = fetchedPosts.reverse().map((post) => {
        // Check if data exists and category property is present
        if (post.data && post.data.description) {
          console.log("Category:", post.data.category); // Log the category for testing
        } else {
          console.error("Missing data or category property");
        }
        return post;
      });
      setPosts(formattedPosts);

      const uniqueCategories = new Set(formattedPosts.map((post) => post.data.category));
      setCategories(['All', ...uniqueCategories]);
    };

    fetchData();
  }, []);

  const filteredPosts = selectedCategory === 'All' ? posts : posts.filter((post) => post.data.category === selectedCategory);

  return (
    <section className="mt-8">
      <BrutalCard colorIndex={1}>
        <div className="text-center">
          <h2 className="text-2xl md:text-4xl lg:text-6xl mb-8 dm-serif">Projects</h2>
        </div>
        <div className='flex justify-center space-x-4 mb-8'>
          {categories.map((category) => (
            <BrutalButton
              key={category}
              variant={selectedCategory === category ? 'default' : 'outline'}
              onClick={() => setSelectedCategory(category)}
            >
              {category}
            </BrutalButton>
          ))}
        </div>


        {/* <ul className='grid md:grid-cols-2 lg:grid-cols-3 gap-8'>
          {filteredPosts.map((post) => (
            <li key={post.id}>
              <BlogSummaryCard post={post} client:load />
            </li>
          ))}
        </ul>
 */}

        {/* <BlogList posts={posts} /> */}

        <ul className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
      {filteredPosts.map((post) => (
        <li key={post.slug}> {/* Add key for better performance */}
          {/* <BlogSummaryCard post={post} /> */}


          <BrutalCard color="white">
      <h3 className='poppins'>{post.data.title}</h3>
      <div className="rounded-lg border-3 border-black my-4 h-56">
        <img src={post.data.imgUrl} alt={post.data.title} className="rounded h-full w-full object-cover" />
      </div>
      <p className='poppins'>{post.data.description}</p>

      <div className="flex justify-end my-4">
        <BrutalButton href={`/blog/${post.slug}/`}>Read post &rarr;</BrutalButton>

      </div>

      <div className="hidden sm:inline-block">
        <p className="poppins mt-2">tags:</p>
        <div className="flex justify-between items-center">
          <ul className="flex gap-4 mt-2">
            {post.data.tags.map((tag) => (
              <li key={tag}> {/* Added key for better performance */}
                <a className="sanchez" href={`/blog/tags/${tag.toLowerCase()}/`}>
                  <BrutalPill>{tag}</BrutalPill>
                </a>
              </li>
            ))}
          </ul>
          {post.data.draft && (
            <span className="bg-green rounded-full border-2 py-1 px-4 text-sm border-black card-shadow">
              Draft
            </span>
          )}
        </div>
      </div>
      {/* {children} */}
    </BrutalCard>

         











        </li>
      ))}
    </ul>
      </BrutalCard>
    </section>
  );
};

export default RecentBlogPosts;
